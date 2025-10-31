"""
Authentication services for the Floor Plan Agent API
"""
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from email_validator import validate_email, EmailNotValidError
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from fastapi import HTTPException

from modules.config.settings import settings
from modules.database import db_manager, User
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

class AuthService:
    """Authentication service class"""
    
    def __init__(self):
        self.db = db_manager

    def _ensure_datetime(self, value) -> Optional[datetime]:
        """Normalize datetime value from database"""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                formats = [
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M:%S.%f",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f"
                ]
                for fmt in formats:
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        continue
        return None

    def _normalize_purpose(self, purpose: str) -> str:
        """Normalize OTP purpose strings from various clients"""
        if not purpose:
            return "email_verification"

        normalized = purpose.strip().lower().replace("-", "_").replace(" ", "_")
        compact = "".join(ch for ch in normalized if ch.isalnum())

        alias_map = {
            "email_verification": "email_verification",
            "emailverification": "email_verification",
            "verify_email": "email_verification",
            "verifyemail": "email_verification",
            "verification": "email_verification",
            "login": "login",
            "signin": "login",
            "sign_in": "login",
            "password_reset": "password_reset",
            "passwordreset": "password_reset",
            "reset_password": "password_reset",
            "resetpassword": "password_reset",
        }

        if normalized in alias_map:
            return alias_map[normalized]
        if compact in alias_map:
            return alias_map[compact]
        return normalized or "email_verification"

    def _sanitize_otp(self, otp: str) -> str:
        """Strip whitespace and formatting characters from OTP input"""
        if otp is None:
            return ""

        cleaned = otp.strip()
        # Remove common formatting characters that may be introduced by UI components
        cleaned = cleaned.replace(" ", "")
        cleaned = cleaned.replace("-", "")
        return cleaned

    def _validate_otp(self, user: User, otp: str, purpose: str):
        """Validate OTP for a user and purpose"""
        normalized_purpose = self._normalize_purpose(purpose)
        otp_record = self.db.get_user_otp(user.id, normalized_purpose)
        if not otp_record or otp_record.otp_code != otp:
            fallback_record = self.db.get_user_otp_by_code(user.id, otp) if otp else None
            if fallback_record:
                # Allow legacy clients that omit the purpose field to continue working
                normalized_purpose = self._normalize_purpose(fallback_record.purpose)
                otp_record = fallback_record
            else:
                raise HTTPException(status_code=400, detail="Invalid or expired OTP")

        if otp_record.consumed_at:
            raise HTTPException(status_code=400, detail="OTP already used")

        expires_at = self._ensure_datetime(otp_record.expires_at)
        if expires_at and expires_at < datetime.now():
            raise HTTPException(status_code=400, detail="OTP has expired")

        return otp_record, normalized_purpose

    def validate_email_format(self, email: str) -> bool:
        """Validate email format"""
        try:
            # Skip deliverability checks for testing
            validate_email(email, check_deliverability=False)
            return True
        except EmailNotValidError:
            return False
    
    def validate_password_strength(self, password: str) -> tuple[bool, str]:
        """Validate password strength"""
        if len(password) < settings.PASSWORD_MIN_LENGTH:
            return False, f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters long"
        return True, ""
    
    def signup_user(self, firstname: str, lastname: str, email: str, password: str, confirm_password: str) -> Dict[str, Any]:
        """Regular signup process"""
        # Validate email format
        if not self.validate_email_format(email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        # Validate password confirmation
        if password != confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match")
        
        # Validate password strength
        is_valid, error_msg = self.validate_password_strength(password)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Check if email already exists
        existing_user = self.db.get_user_by_email(email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already exists")
        
        # Create new user
        try:
            user_id = self.db.create_user(firstname, lastname, email, password)
            
            # Send verification email
            from modules.auth.email_service import email_service
            email_sent = email_service.send_verification_email(user_id, email, firstname)
            
            return {
                "message": "User created successfully. Please check your email to verify your account.",
                "user_id": user_id,
                "firstname": firstname,
                "lastname": lastname,
                "email": email,
                "verification_email_sent": email_sent,
                "requires_verification": True,
                "otp_purpose": "email_verification",
                "otp_delivery": "email"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    def login_user(self, email: str, password: str) -> Dict[str, Any]:
        """Regular login process with email verification check"""
        # Validate email format
        if not self.validate_email_format(email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        # Verify credentials
        user = self.db.verify_user_credentials(email, password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Check if email is verified (except for Google OAuth users)
        if not user.is_verified and not user.google_id:
            # Automatically resend verification email
            from modules.auth.email_service import email_service
            email_sent = email_service.resend_verification_email(user.id, user.email, user.firstname)

            return {
                "message": "Please verify your email address before logging in. We've sent a new verification email.",
                "user_id": user.id,
                "email": user.email,
                "requires_verification": True,
                "verified": False,
                "verification_email_sent": email_sent,
                "otp_purpose": "email_verification",
                "otp_delivery": "email"
            }

        from modules.auth.email_service import email_service
        otp_sent = email_service.send_login_otp_email(
            user.id,
            user.email,
            user.firstname,
            provider="google" if user.google_id else "password"
        )

        return {
            "message": "OTP sent to your email. Please verify to complete login.",
            "user_id": user.id,
            "firstname": user.firstname,
            "lastname": user.lastname,
            "email": user.email,
            "verified": user.is_verified,
            "requires_otp": True,
            "otp_delivery": "email",
            "otp_purpose": "login",
            "otp_sent": otp_sent
        }
    
    def google_signup(self, id_token_str: str) -> Dict[str, Any]:
        """Google OAuth signup process"""
        print(f"DEBUG: Google signup called with token: {id_token_str[:50]}...")
        print(f"DEBUG: GOOGLE_CLIENT_ID configured: {bool(settings.GOOGLE_CLIENT_ID)}")
        
        if not settings.GOOGLE_CLIENT_ID:
            raise HTTPException(status_code=500, detail="Google OAuth not configured")
        
        # Validate that the token looks like a JWT (starts with 'eyJ')
        if not id_token_str.startswith('eyJ'):
            raise HTTPException(status_code=400, detail="Invalid ID token format. Expected JWT token starting with 'eyJ'")
        
        try:
            # Verify the Google ID token
            print(f"DEBUG: Attempting to verify token with client ID: {settings.GOOGLE_CLIENT_ID[:20]}...")
            idinfo = id_token.verify_oauth2_token(id_token_str, google_requests.Request(), settings.GOOGLE_CLIENT_ID)
            print(f"DEBUG: Token verified successfully. User info: {idinfo.get('email', 'no_email')}")
            
            # Extract user information from Google
            google_user_id = idinfo['sub']
            email = idinfo['email']
            firstname = idinfo.get('given_name', '')
            lastname = idinfo.get('family_name', '')
            
            print(f"DEBUG: Extracted info - email: {email}, name: {firstname} {lastname}")
            
            # Check if user already exists with this email
            existing_user = self.db.get_user_by_email(email)
            if existing_user:
                raise HTTPException(status_code=400, detail="User with this email already exists")
            
            # Check if Google ID already exists
            existing_google_user = self.db.get_user_by_google_id(google_user_id)
            if existing_google_user:
                raise HTTPException(status_code=400, detail="Google account already linked to another user")
            
            # Create new user with Google OAuth
            # Generate a random password since they're using Google OAuth
            random_password = secrets.token_urlsafe(32)
            
            user_id = self.db.create_user(firstname, lastname, email, random_password, google_user_id)
            
            # Mark Google OAuth users as verified by default
            self.db.verify_user_email(user_id)

            from modules.auth.email_service import email_service
            otp_sent = email_service.send_login_otp_email(user_id, email, firstname, provider="google")

            return {
                "message": "User created successfully. Please confirm the OTP sent to your email to continue.",
                "user_id": user_id,
                "firstname": firstname,
                "lastname": lastname,
                "email": email,
                "verified": True,
                "action": "signup",
                "requires_otp": True,
                "otp_delivery": "email",
                "otp_purpose": "login",
                "otp_sent": otp_sent
            }
            
        except ValueError as e:
            # Invalid token
            print(f"DEBUG: Token validation failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid Google ID token: {str(e)}")
        except Exception as e:
            print(f"DEBUG: General error in google_signup: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Google signup error: {str(e)}")
    
    def google_signin(self, id_token_str: str) -> Dict[str, Any]:
        """Google OAuth signin process"""
        print(f"DEBUG: Google signin called with token: {id_token_str[:50]}...")
        print(f"DEBUG: GOOGLE_CLIENT_ID configured: {bool(settings.GOOGLE_CLIENT_ID)}")
        
        if not settings.GOOGLE_CLIENT_ID:
            raise HTTPException(status_code=500, detail="Google OAuth not configured")
        
        # Validate that the token looks like a JWT (starts with 'eyJ')
        if not id_token_str.startswith('eyJ'):
            raise HTTPException(status_code=400, detail="Invalid ID token format. Expected JWT token starting with 'eyJ'")
        
        try:
            # Verify the Google ID token
            print(f"DEBUG: Attempting to verify token with client ID: {settings.GOOGLE_CLIENT_ID[:20]}...")
            idinfo = id_token.verify_oauth2_token(id_token_str, google_requests.Request(), settings.GOOGLE_CLIENT_ID)
            print(f"DEBUG: Token verified successfully. User info: {idinfo.get('email', 'no_email')}")
            
            # Extract user information from Google
            google_user_id = idinfo['sub']
            email = idinfo['email']
            
            # Find user by Google ID first, then by email
            user = self.db.get_user_by_google_id(google_user_id)
            if not user:
                user = self.db.get_user_by_email(email)
            
            if not user:
                raise HTTPException(status_code=404, detail="User not found. Please sign up first.")
            
            # If user exists but doesn't have Google ID linked, link it
            if user.google_id is None:
                self.db.update_user_google_id(user.id, google_user_id)
            
            from modules.auth.email_service import email_service
            otp_sent = email_service.send_login_otp_email(user.id, user.email, user.firstname, provider="google")

            return {
                "message": "OTP sent to your email. Please verify to complete login.",
                "user_id": user.id,
                "firstname": user.firstname,
                "lastname": user.lastname,
                "email": user.email,
                "requires_otp": True,
                "otp_delivery": "email",
                "otp_purpose": "login",
                "otp_sent": otp_sent
            }
            
        except ValueError as e:
            # Invalid token
            print(f"DEBUG: Token validation failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid Google ID token: {str(e)}")
        except Exception as e:
            print(f"DEBUG: General error in google_signin: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Google signin error: {str(e)}")
    
    def google_signin_userinfo(self, email: str, google_id: str) -> Dict[str, Any]:
        """Google OAuth signin process using user info"""
        print(f"DEBUG: Google signin userinfo called for email: {email}, google_id: {google_id[:20]}...")
        
        try:
            # Find user by Google ID first, then by email
            user = self.db.get_user_by_google_id(google_id)
            if not user:
                user = self.db.get_user_by_email(email)
            
            if not user:
                raise HTTPException(status_code=404, detail="User not found. Please sign up first.")
            
            # If user exists but doesn't have Google ID linked, link it
            if user.google_id is None:
                self.db.update_user_google_id(user.id, google_id)
            
            from modules.auth.email_service import email_service
            otp_sent = email_service.send_login_otp_email(user.id, user.email, user.firstname, provider="google")

            return {
                "message": "OTP sent to your email. Please verify to complete login.",
                "user_id": user.id,
                "firstname": user.firstname,
                "lastname": user.lastname,
                "email": user.email,
                "requires_otp": True,
                "otp_delivery": "email",
                "otp_purpose": "login",
                "otp_sent": otp_sent
            }
            
        except Exception as e:
            print(f"DEBUG: General error in google_signin_userinfo: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Google signin error: {str(e)}")
    
    def google_signup_userinfo(self, email: str, firstname: str, lastname: str, google_id: str) -> Dict[str, Any]:
        """Google OAuth signup process using user info"""
        print(f"DEBUG: Google signup userinfo called for email: {email}, name: {firstname} {lastname}")
        
        try:
            # Validate email format
            if not self.validate_email_format(email):
                raise HTTPException(status_code=400, detail="Invalid email format")
            
            # Check if user already exists with this email
            existing_user = self.db.get_user_by_email(email)
            if existing_user:
                raise HTTPException(status_code=400, detail="User with this email already exists")
            
            # Check if Google ID already exists
            existing_google_user = self.db.get_user_by_google_id(google_id)
            if existing_google_user:
                raise HTTPException(status_code=400, detail="Google account already linked to another user")
            
            # Create new user with Google OAuth
            # Generate a random password since they're using Google OAuth
            random_password = secrets.token_urlsafe(32)
            
            user_id = self.db.create_user(firstname, lastname, email, random_password, google_id)

            self.db.verify_user_email(user_id)

            from modules.auth.email_service import email_service
            otp_sent = email_service.send_login_otp_email(user_id, email, firstname, provider="google")

            return {
                "message": "User created successfully. Please confirm the OTP sent to your email to continue.",
                "user_id": user_id,
                "firstname": firstname,
                "lastname": lastname,
                "email": email,
                "requires_otp": True,
                "otp_delivery": "email",
                "otp_purpose": "login",
                "otp_sent": otp_sent
            }
            
        except Exception as e:
            print(f"DEBUG: General error in google_signup_userinfo: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Google signup error: {str(e)}")
    
    def verify_email(self, token: str) -> Dict[str, Any]:
        """Verify user email using verification token"""
        try:
            # Find user by verification token
            user = self.db.get_user_by_verification_token(token)
            if not user:
                raise HTTPException(status_code=400, detail="Invalid or expired verification token")
            
            # Check if token is expired
            if user.verification_token_expires and user.verification_token_expires < datetime.now():
                raise HTTPException(status_code=400, detail="Verification token has expired")
            
            # Check if already verified
            if user.is_verified:
                return {
                    "message": "Email already verified",
                    "user_id": user.id,
                    "firstname": user.firstname,
                    "lastname": user.lastname,
                    "email": user.email,
                    "already_verified": True
                }
            
            # Mark user as verified
            self.db.verify_user_email(user.id)

            # Consume OTP if one exists for email verification
            if user.verification_token:
                try:
                    self.db.consume_user_otp(user.id, user.verification_token, "email_verification")
                except Exception:
                    pass

            # Send welcome email
            from modules.auth.email_service import email_service
            email_service.send_verification_success_email(user.email, user.firstname)
            
            return {
                "message": "Email verified successfully! You can now log in.",
                "user_id": user.id,
                "email": user.email,
                "verified": True
            }
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"DEBUG: Error in verify_email: {str(e)}")
            raise HTTPException(status_code=500, detail="Email verification error")
    
    def resend_verification_email(self, email: str) -> Dict[str, Any]:
        """Resend verification email to user"""
        try:
            # Find user by email
            user = self.db.get_user_by_email(email)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Check if already verified
            if user.is_verified:
                return {
                    "message": "Email is already verified",
                    "email": email,
                    "already_verified": True
                }
            
            # Send verification email
            from modules.auth.email_service import email_service
            email_sent = email_service.resend_verification_email(user.id, email, user.firstname)
            
            return {
                "message": "Verification email sent successfully" if email_sent else "Verification email could not be sent (check server configuration)",
                "email": email,
                "email_sent": email_sent,
                "otp_delivery": "email",
                "otp_purpose": "email_verification"
            }

        except HTTPException:
            raise

    def forgot_password(self, email: str) -> Dict[str, Any]:
        """Send password reset OTP"""
        user = self.db.get_user_by_email(email)

        if not user:
            return {
                "message": "If an account exists for this email, a reset code has been sent.",
                "email": email
            }

        from modules.auth.email_service import email_service
        email_sent = email_service.send_password_reset_email(user.id, user.email, user.firstname)

        return {
            "message": "Password reset code sent to your email.",
            "email": user.email,
            "otp_delivery": "email",
            "otp_purpose": "password_reset",
            "otp_sent": email_sent
        }

    def reset_password(self, email: str, otp: str, new_password: str, confirm_password: str) -> Dict[str, Any]:
        """Reset user password after OTP validation"""
        user = self.db.get_user_by_email(email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if new_password != confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match")

        is_valid, error_msg = self.validate_password_strength(new_password)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        sanitized_otp = self._sanitize_otp(otp)
        _, normalized_purpose = self._validate_otp(user, sanitized_otp, "password_reset")
        self.db.consume_user_otp(user.id, sanitized_otp, normalized_purpose)

        if not self.db.update_user_password(user.id, new_password):
            raise HTTPException(status_code=500, detail="Failed to update password")

        return {
            "message": "Password reset successfully",
            "user_id": user.id,
            "email": user.email
        }

    def verify_otp(self, email: str, otp: str, purpose: str = "email_verification") -> Dict[str, Any]:
        """Verify OTP for a specific purpose"""
        try:
            sanitized_email = email.strip() if email else email
            sanitized_otp = self._sanitize_otp(otp)

            user = self.db.get_user_by_email(sanitized_email)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            _otp_record, normalized_purpose = self._validate_otp(user, sanitized_otp, purpose)

            # Mark OTP as consumed for single-step flows only. For password reset we
            # allow a separate reset call to consume the OTP after verification so
            # that users can first confirm the code and then submit their new
            # password without getting an "OTP already used" error.
            if normalized_purpose != "password_reset":
                self.db.consume_user_otp(user.id, sanitized_otp, normalized_purpose)

            if normalized_purpose == "email_verification":
                if user.is_verified:
                    return {
                        "message": "Email already verified",
                        "user_id": user.id,
                        "firstname": user.firstname,
                        "lastname": user.lastname,
                        "email": user.email,
                        "already_verified": True
                    }

                self.db.verify_user_email(user.id)

                from modules.auth.email_service import email_service
                email_service.send_verification_success_email(user.email, user.firstname)

                return {
                    "message": "Email verified successfully! You can now log in.",
                    "user_id": user.id,
                    "firstname": user.firstname,
                    "lastname": user.lastname,
                    "email": user.email,
                    "verified": True
                }

            if normalized_purpose == "login":
                self.db.update_user_last_login(user.id)
                self.db.log_user_activity(user.id, "login", {"email": user.email})
                access_token = self.generate_token(user.id, user.email)

                return {
                    "message": "Login successful",
                    "user_id": user.id,
                    "firstname": user.firstname,
                    "lastname": user.lastname,
                    "email": user.email,
                    "verified": user.is_verified,
                    "otp_verified": True,
                    "access_token": access_token,
                    "token_type": "bearer"
                }

            if normalized_purpose == "password_reset":
                return {
                    "message": "OTP verified",
                    "user_id": user.id,
                    "email": user.email,
                    "otp_verified": True
                }

            return {
                "message": "OTP verified",
                "user_id": user.id,
                "email": user.email,
                "otp_verified": True,
                "purpose": normalized_purpose
            }

        except HTTPException:
            raise
        except Exception as e:
            print(f"DEBUG: Error in verify_otp: {str(e)}")
            raise HTTPException(status_code=500, detail="OTP verification error")
    
    def continue_with_google(self, id_token_str: str) -> Dict[str, Any]:
        """Google OAuth continue process - handles both signup and signin automatically"""
        print(f"DEBUG: Continue with Google called with token: {id_token_str[:50]}...")
        print(f"DEBUG: GOOGLE_CLIENT_ID configured: {bool(settings.GOOGLE_CLIENT_ID)}")
        
        if not settings.GOOGLE_CLIENT_ID:
            raise HTTPException(status_code=500, detail="Google OAuth not configured")
        
        # Validate that the token looks like a JWT (starts with 'eyJ')
        if not id_token_str.startswith('eyJ'):
            raise HTTPException(status_code=400, detail="Invalid ID token format. Expected JWT token starting with 'eyJ'")
        
        try:
            # Verify the Google ID token
            print(f"DEBUG: Attempting to verify token with client ID: {settings.GOOGLE_CLIENT_ID[:20]}...")
            idinfo = id_token.verify_oauth2_token(id_token_str, google_requests.Request(), settings.GOOGLE_CLIENT_ID)
            print(f"DEBUG: Token verified successfully. User info: {idinfo.get('email', 'no_email')}")
            
            # Extract user information from Google
            google_user_id = idinfo['sub']
            email = idinfo['email']
            firstname = idinfo.get('given_name', '')
            lastname = idinfo.get('family_name', '')
            
            print(f"DEBUG: Extracted info - email: {email}, name: {firstname} {lastname}")
            
            # First, try to find user by Google ID
            existing_user = self.db.get_user_by_google_id(google_user_id)
            
            if existing_user:
                # User exists with this Google ID - sign them in
                print(f"DEBUG: User found by Google ID - signing in")
                from modules.auth.email_service import email_service
                otp_sent = email_service.send_login_otp_email(existing_user.id, existing_user.email, existing_user.firstname, provider="google")

                return {
                    "message": "OTP sent to your email. Please verify to continue.",
                    "user_id": existing_user.id,
                    "firstname": existing_user.firstname,
                    "lastname": existing_user.lastname,
                    "email": existing_user.email,
                    "action": "signin",
                    "requires_otp": True,
                    "otp_delivery": "email",
                    "otp_purpose": "login",
                    "otp_sent": otp_sent
                }
            
            # If not found by Google ID, try to find by email
            existing_user = self.db.get_user_by_email(email)
            
            if existing_user:
                # User exists with this email but no Google ID linked
                if existing_user.google_id is None:
                    # Link the Google ID to existing account and sign them in
                    print(f"DEBUG: User found by email, linking Google ID")
                    self.db.update_user_google_id(existing_user.id, google_user_id)
                    from modules.auth.email_service import email_service
                    otp_sent = email_service.send_login_otp_email(existing_user.id, existing_user.email, existing_user.firstname, provider="google")

                    return {
                        "message": "Account linked successfully. Please verify the OTP to continue.",
                        "user_id": existing_user.id,
                        "firstname": existing_user.firstname,
                        "lastname": existing_user.lastname,
                        "email": existing_user.email,
                        "action": "signin_and_link",
                        "requires_otp": True,
                        "otp_delivery": "email",
                        "otp_purpose": "login",
                        "otp_sent": otp_sent
                    }
                else:
                    # User has different Google ID - this shouldn't happen but handle it
                    print(f"DEBUG: User has different Google ID linked")
                    raise HTTPException(
                        status_code=400, 
                        detail="This email is already associated with a different Google account"
                    )
            
            # User doesn't exist - create new account (signup)
            print(f"DEBUG: User not found - creating new account")
            
            # Generate a random password since they're using Google OAuth
            random_password = secrets.token_urlsafe(32)
            
            user_id = self.db.create_user(firstname, lastname, email, random_password, google_user_id)
            
            # Mark Google OAuth users as verified by default
            self.db.verify_user_email(user_id)
            
            from modules.auth.email_service import email_service
            otp_sent = email_service.send_login_otp_email(user_id, email, firstname, provider="google")

            return {
                "message": "Welcome! Account created successfully. Confirm the OTP sent to your email to continue.",
                "user_id": user_id,
                "firstname": firstname,
                "lastname": lastname,
                "email": email,
                "verified": True,
                "action": "signup",
                "requires_otp": True,
                "otp_delivery": "email",
                "otp_purpose": "login",
                "otp_sent": otp_sent
            }
            
        except ValueError as e:
            # Invalid token
            print(f"DEBUG: Token validation failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid Google ID token: {str(e)}")
        except Exception as e:
            print(f"DEBUG: General error in continue_with_google: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Google authentication error: {str(e)}")
    # Add these methods to your existing AuthService class



    def generate_token(self, user_id: int, email: str) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user_id,
            "email": email,
            "exp": datetime.utcnow() + timedelta(days=7)  # 7 days expiry
        }
        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")

    def verify_token(self, token: str) -> Optional[int]:
        """Verify JWT token and return user ID"""
        try:
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS256"])
            return payload.get("user_id")
        except ExpiredSignatureError:
            return None
        except InvalidTokenError:
            return None

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        return self.db.get_user_by_id(user_id)

    def update_user_profile(self, user_id: int, **kwargs) -> bool:
        """Update user profile"""
        return self.db.update_user_profile(user_id, **kwargs)
# Global auth service instance
auth_service = AuthService()