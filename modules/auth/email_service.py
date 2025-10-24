"""
Email service for sending verification and notification emails
"""
import smtplib
import uuid
import random
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List
from modules.config.settings import settings
from modules.database import db_manager

class EmailService:
    """Email service for sending verification emails"""
    
    def __init__(self):
        self.smtp_server = settings.SMTP_SERVER
        self.smtp_port = settings.SMTP_PORT
        self.smtp_username = settings.SMTP_USERNAME
        self.smtp_password = settings.SMTP_PASSWORD
        self.from_email = settings.FROM_EMAIL
        self.frontend_url = settings.FRONTEND_URL
    
    def is_configured(self) -> bool:
        """Check if email service is properly configured"""
        return bool(self.smtp_server and self.smtp_username and self.smtp_password)
    
    def generate_verification_token(self) -> str:
        """Generate a 6-digit OTP code"""
        return str(random.randint(100000, 999999))

    def _normalize_purpose(self, purpose: str) -> str:
        """Normalize OTP purpose so it aligns with the authentication service"""
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

    def _compose_otp_email(
        self,
        firstname: str,
        heading: str,
        intro_text: str,
        otp_code: str,
        expiry_minutes: int,
        instructions: List[str],
        footer_note: Optional[str] = None
    ) -> tuple[str, str]:
        """Build HTML and text email bodies for OTP messages"""
        instructions_html = "".join(f"<li>{step}</li>" for step in instructions)
        footer_section = f"<p>{footer_note}</p>" if footer_note else ""

        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>{heading}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }}
        .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
        .otp-code {{ background: #667eea; color: white; font-size: 32px; font-weight: bold; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0; letter-spacing: 8px; font-family: monospace; }}
        .footer {{ text-align: center; margin-top: 30px; font-size: 12px; color: #666; }}
        .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class=\"container\">
        <div class=\"header\">
            <h1>{heading}</h1>
                <p>Secure access with Ecadoc</p>
        </div>
        <div class=\"content\">
            <h2>Hi {firstname}!</h2>
            <p>{intro_text}</p>
            <div class=\"otp-code\">{otp_code}</div>
            <div class=\"warning\">
                <strong>⚠️ Important:</strong> This code will expire in <strong>{expiry_minutes} minutes</strong>.
            </div>
            <p><strong>Next steps:</strong></p>
            <ol>{instructions_html}</ol>
            {footer_section}
        </div>
        <div class=\"footer\">
                <p>This email was sent by Ecadoc. If you have questions, please contact our support team.</p>
            <p>Code expires at: {{expires_at}}</p>
        </div>
    </div>
</body>
</html>
        """

        text_lines = [
            f"Hi {firstname}!",
            "",
            intro_text,
            "",
            f"Verification Code: {otp_code}",
            "",
            f"This code will expire in {expiry_minutes} minutes.",
            "",
            "Next steps:",
        ]

        for idx, step in enumerate(instructions, start=1):
            text_lines.append(f"{idx}. {step}")

        if footer_note:
            text_lines.extend(["", footer_note])

        text_lines.append("")
        text_lines.append("Ecadoc Team")

        text_body = "\n".join(text_lines)
        return html_body, text_body

    def _send_otp_email(
        self,
        user_id: int,
        email: str,
        firstname: str,
        purpose: str,
        subject: str,
        heading: str,
        intro_text: str,
        instructions: List[str],
        expiry_minutes: int = 5,
        footer_note: Optional[str] = None
    ) -> bool:
        """Send OTP email with shared layout"""
        normalized_purpose = self._normalize_purpose(purpose)
        otp_code = self.generate_verification_token()
        expires_at = datetime.now() + timedelta(minutes=expiry_minutes)

        db_manager.create_verification_token(user_id, otp_code, expires_at, purpose=normalized_purpose)

        if not self.is_configured():
            print("WARNING: Email service not configured - OTP email not sent")
            print(
                f"DEBUG: Generated OTP {otp_code} for user {user_id} ({normalized_purpose}), "
                f"expires at {expires_at.isoformat()}"
            )
            return False

        html_body, text_body = self._compose_otp_email(
            firstname=firstname,
            heading=heading,
            intro_text=intro_text,
            otp_code=otp_code,
            expiry_minutes=expiry_minutes,
            instructions=instructions,
            footer_note=footer_note
        )

        html_body = html_body.replace("{{expires_at}}", expires_at.strftime('%Y-%m-%d %H:%M:%S'))

        return self._send_email(email, subject, text_body, html_body)

    def send_verification_email(self, user_id: int, email: str, firstname: str) -> bool:
        """Send email verification email to user"""
        try:
            return self._send_otp_email(
                user_id=user_id,
                email=email,
                firstname=firstname,
                purpose="email_verification",
                    subject="Your Verification Code - Ecadoc",
                heading="Verify Your Email",
                intro_text="Thank you for signing up for Esticore. Please use the verification code below to complete your registration.",
                    instructions=[
                        "Return to the Ecadoc application",
                    "Enter the 6-digit code above",
                    "Complete your account verification"
                ],
                footer_note="If you didn't create this account, you can safely ignore this email."
            )
        except Exception as e:
            print(f"ERROR: Failed to send verification email: {e}")
            return False

    def send_login_otp_email(self, user_id: int, email: str, firstname: str, provider: str = "password") -> bool:
        """Send OTP email for login confirmation"""
        intro = "We noticed a login attempt to your Ecadoc account. Enter the code below to confirm it's you."
        if provider == "google":
                intro = "We noticed a Google sign-in attempt to your Ecadoc account. Enter the code below to confirm it's you."

        return self._send_otp_email(
            user_id=user_id,
            email=email,
            firstname=firstname,
            purpose="login",
                subject="Your Login Code - Ecadoc",
            heading="Confirm Your Login",
            intro_text=intro,
            instructions=[
                    "Return to the Ecadoc login screen",
                "Enter the 6-digit code",
                "Continue to your workspace"
            ]
        )

    def send_password_reset_email(self, user_id: int, email: str, firstname: str) -> bool:
        """Send OTP email for password reset"""
        return self._send_otp_email(
            user_id=user_id,
            email=email,
            firstname=firstname,
            purpose="password_reset",
                subject="Reset Your Password - Ecadoc",
            heading="Password Reset Request",
                intro_text="A password reset was requested for your Ecadoc account. Use the code below to continue.",
            instructions=[
                "Return to the password reset screen",
                "Enter the 6-digit code",
                "Create a new secure password"
            ],
            footer_note="If you did not request this, please ignore the email."
        )

    def send_project_invitation_email(
        self,
        invitee_email: str,
        invitee_name: str,
        inviter_name: str,
        project_name: str,
        action_url: Optional[str] = None
    ) -> bool:
        """Send project invitation email"""
        if not self.is_configured():
            print("WARNING: Email service not configured - project invitation email not sent")
            return False

        subject = f"{inviter_name} invited you to collaborate on '{project_name}'"
        action_button = ""
        if action_url:
            action_button = f"<p style=\"text-align:center;\"><a href=\"{action_url}\" style=\"display:inline-block;padding:12px 30px;background:#667eea;color:#fff;text-decoration:none;border-radius:5px;\">View invitation</a></p>"

        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Project Invitation</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }}
        .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
    </style>
</head>
<body>
    <div class=\"container\">
        <div class=\"header\">
            <h1>You've been invited!</h1>
        </div>
        <div class=\"content\">
            <p>Hi {invitee_name},</p>
                <p><strong>{inviter_name}</strong> invited you to collaborate on the project <strong>{project_name}</strong> in Ecadoc.</p>
                <p>You can accept or decline the invitation directly from the notifications panel inside Ecadoc.</p>
            {action_button}
            <p>If you weren't expecting this, you can safely ignore the email.</p>
        </div>
    </div>
</body>
</html>
        """

        text_body = f"""Hi {invitee_name},

{inviter_name} invited you to collaborate on the project '{project_name}' in Esticore.
You can accept or decline the invitation from the notifications panel inside Esticore.

If you weren't expecting this, you can ignore this email.
"""

        return self._send_email(invitee_email, subject, text_body, html_body)

    
    def send_verification_success_email(self, email: str, firstname: str) -> bool:
        """Send confirmation email after successful verification"""
        if not self.is_configured():
            return False
        
        try:
            subject = "Email Verified Successfully - Welcome to Esticore!"
            
            html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Email Verified</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }}
        .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
        .success {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .button {{ display: inline-block; padding: 12px 30px; background: #28a745; color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎉 Email Verified!</h1>
            <p>Your account is now active</p>
        </div>
        <div class="content">
            <div class="success">
                <strong>Success!</strong> Your email address has been verified successfully.
            </div>
            
            <h2>Hi {firstname}!</h2>
                <p>Congratulations! Your email has been verified and your Ecadoc account is now fully active.</p>
            
            <p style="text-align: center;">
                    <a href="{self.frontend_url}/login" class="button">Start Using Ecadoc</a>
            </p>
            
            <p><strong>What you can do now:</strong></p>
            <ul>
                <li>Upload and process floor plan documents</li>
                <li>Use AI-powered annotation tools</li>
                <li>Create and manage projects</li>
                <li>Access all premium features</li>
            </ul>
            
            <p>Welcome aboard! We're excited to help you with your floor plan analysis needs.</p>
        </div>
    </div>
</body>
</html>
            """
            
            text_body = f"""
Hi {firstname}!

🎉 Your email has been verified successfully!

Your Esticore account is now fully active and ready to use.

What you can do now:

Welcome aboard! We're excited to help you with your floor plan analysis needs.

Login at: {self.frontend_url}/login

Esticore Team
            """
            
            return self._send_email(email, subject, text_body, html_body)
            
        except Exception as e:
            print(f"ERROR: Failed to send verification success email: {e}")
            return False
    
    def resend_verification_email(self, user_id: int, email: str, firstname: str) -> bool:
        """Resend verification email (generates new token)"""
        return self.send_verification_email(user_id, email, firstname)
    
    def _send_email(self, to_email: str, subject: str, text_body: str, html_body: str = None) -> bool:
        """Send email using SMTP"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"Ecadoc <{self.from_email}>"
            msg['To'] = to_email
            
            # Attach parts
            part1 = MIMEText(text_body, 'plain')
            msg.attach(part1)
            
            if html_body:
                part2 = MIMEText(html_body, 'html')
                msg.attach(part2)
            
            # Connect and send
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            print(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            print(f"SMTP Error: Failed to send email to {to_email}: {e}")
            return False

# Global email service instance
email_service = EmailService()