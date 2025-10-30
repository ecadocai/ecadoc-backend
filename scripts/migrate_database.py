"""
Database migration script for admin and subscription features
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.database import db_manager
from modules.config.settings import settings

def migrate_database():
    """Run database migrations for admin and subscription features"""
    conn = db_manager.get_connection()
    cur = conn.cursor()
    
    try:
        print("Starting database migration...")
        
        # Check if we're using MySQL or SQLite
        use_rds = settings.USE_RDS
        
        # 1. Create admin_users table (PostgreSQL syntax)
        print("Creating admin_users table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS admin_users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                is_super_admin BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_admin_email ON admin_users (email)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_admin_super_admin ON admin_users (is_super_admin)")
        
        # 2. Create subscription_plans table (PostgreSQL syntax)
        print("Creating subscription_plans table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS subscription_plans (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL,
                description TEXT,
                storage_gb INT DEFAULT 0,
                project_limit INT DEFAULT 0,
                user_limit INT DEFAULT 1,
                action_limit INT DEFAULT 0,
                features JSON,
                is_active BOOLEAN DEFAULT TRUE,
                has_free_trial BOOLEAN DEFAULT FALSE,
                trial_days INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_plan_name ON subscription_plans (name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_plan_active ON subscription_plans (is_active)")

        # 2b. Create subscription_plan_prices table
        print("Ensuring subscription_plan_prices table exists...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS subscription_plan_prices (
                id SERIAL PRIMARY KEY,
                plan_id INT NOT NULL REFERENCES subscription_plans (id) ON DELETE CASCADE,
                duration_months INT NOT NULL,
                price NUMERIC(10,2) NOT NULL,
                currency VARCHAR(10) DEFAULT 'usd',
                stripe_price_id VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (plan_id, duration_months)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_plan_prices_plan ON subscription_plan_prices (plan_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_plan_prices_duration ON subscription_plan_prices (duration_months)")
        
        # 3. Create user_subscriptions table
        print("Creating user_subscriptions table...")
        # 3. Create user_subscriptions table (PostgreSQL syntax)
        print("Creating user_subscriptions table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_subscriptions (
                id SERIAL PRIMARY KEY,
                user_id INT NOT NULL,
                plan_id INT NOT NULL,
                stripe_subscription_id VARCHAR(255),
                stripe_customer_id VARCHAR(255),
                current_period_start TIMESTAMP,
                current_period_end TIMESTAMP,
                status VARCHAR(50) DEFAULT 'active',
                interval VARCHAR(20) DEFAULT 'six_month',
                auto_renew BOOLEAN DEFAULT TRUE,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                FOREIGN KEY (plan_id) REFERENCES subscription_plans (id)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sub_user_id ON user_subscriptions (user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sub_plan_id ON user_subscriptions (plan_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sub_status ON user_subscriptions (status)")

        # 3b. Migrate legacy subscription pricing if columns still exist
        print("Checking for legacy pricing columns...")
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name='subscription_plans' AND column_name IN ('price_quarterly', 'price_annual')
        """)
        legacy_columns = {row[0] for row in cur.fetchall()}

        if legacy_columns:
            print("Migrating legacy pricing columns into subscription_plan_prices...")
            select_columns = ["id"]
            if 'price_quarterly' in legacy_columns:
                select_columns.append("price_quarterly")
            else:
                select_columns.append("NULL AS price_quarterly")

            if 'price_annual' in legacy_columns:
                select_columns.append("price_annual")
            else:
                select_columns.append("NULL AS price_annual")

            cur.execute(f"SELECT {', '.join(select_columns)} FROM subscription_plans")
            rows = cur.fetchall()
            for plan_id, quarterly_price, annual_price in rows:
                if quarterly_price and float(quarterly_price) > 0:
                    six_month_price = float(quarterly_price) * 2
                    cur.execute(
                        """
                            INSERT INTO subscription_plan_prices (plan_id, duration_months, price, currency)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (plan_id, duration_months) DO NOTHING
                        """,
                        (plan_id, 6, six_month_price, 'usd'),
                    )
                if annual_price and float(annual_price) > 0:
                    cur.execute(
                        """
                            INSERT INTO subscription_plan_prices (plan_id, duration_months, price, currency)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (plan_id, duration_months) DO NOTHING
                        """,
                        (plan_id, 12, float(annual_price), 'usd'),
                    )

            if 'price_quarterly' in legacy_columns:
                print("Dropping price_quarterly column...")
                cur.execute("ALTER TABLE subscription_plans DROP COLUMN price_quarterly")
            if 'price_annual' in legacy_columns:
                print("Dropping price_annual column...")
                cur.execute("ALTER TABLE subscription_plans DROP COLUMN price_annual")

        # 3c. Normalize legacy interval values
        print("Normalizing legacy subscription intervals...")
        cur.execute("ALTER TABLE user_subscriptions ALTER COLUMN interval SET DEFAULT 'six_month'")
        cur.execute("""
            UPDATE user_subscriptions
            SET interval = 'six_month'
            WHERE interval IN ('quarterly', 'quarter', 'monthly', 'month')
        """)
        
        # 4. Create user_storage table
        print("Creating user_storage table...")
        # 4. Create user_storage table (PostgreSQL syntax)
        print("Creating user_storage table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_storage (
                id SERIAL PRIMARY KEY,
                user_id INT UNIQUE NOT NULL,
                used_storage_mb INT DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_storage_user_id ON user_storage (user_id)")
        
        # 5. Create feedback table
        print("Creating feedback table...")
        # 5. Create feedback table (PostgreSQL syntax)
        print("Creating feedback table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                user_id INT NOT NULL,
                email VARCHAR(255) NOT NULL,
                ai_response TEXT NOT NULL,
                rating VARCHAR(8) CHECK(rating IN ('positive', 'negative')) NOT NULL,
                project_name VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback (user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback (rating)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback (created_at)")
        
        # 6. Create ai_models table
        print("Creating ai_models table...")
        # 6. Create ai_models table (PostgreSQL syntax)
        print("Creating ai_models table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ai_models (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL,
                provider VARCHAR(100) NOT NULL,
                model_name VARCHAR(255) NOT NULL,
                is_active BOOLEAN DEFAULT FALSE,
                config JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ai_provider ON ai_models (provider)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ai_active ON ai_models (is_active)")
        
        # 7. Create recently_viewed_projects table
        print("Creating recently_viewed_projects table...")
        # 7. Create recently_viewed_projects table (PostgreSQL syntax)
        print("Creating recently_viewed_projects table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS recently_viewed_projects (
                id SERIAL PRIMARY KEY,
                user_id INT NOT NULL,
                project_id VARCHAR(255) NOT NULL,
                viewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                view_count INT DEFAULT 1,
                UNIQUE (user_id, project_id),
                FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE CASCADE
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_recent_user_id ON recently_viewed_projects (user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_recent_viewed_at ON recently_viewed_projects (viewed_at)")
        
        # 8. Add is_active column to userdata table if it doesn't exist (PostgreSQL)
        print("Checking userdata table for is_active column...")
        cur.execute("""
            SELECT column_name FROM information_schema.columns WHERE table_name='userdata' AND column_name='is_active'
        """)
        column_exists = cur.fetchone() is not None
        if not column_exists:
            print("Adding is_active column to userdata table...")
            cur.execute("ALTER TABLE userdata ADD COLUMN is_active BOOLEAN DEFAULT TRUE")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_active ON userdata (is_active)")
        
        # 9. Add profile_image column to userdata table if it doesn't exist (PostgreSQL)
        print("Checking userdata table for profile_image column...")
        cur.execute("""
            SELECT column_name FROM information_schema.columns WHERE table_name='userdata' AND column_name='profile_image'
        """)
        column_exists = cur.fetchone() is not None
        if not column_exists:
            print("Adding profile_image column to userdata table...")
            cur.execute("ALTER TABLE userdata ADD COLUMN profile_image TEXT")
        
        # 10. Create default admin user if no admin exists (PostgreSQL)
        print("Creating default admin user...")
        cur.execute("SELECT COUNT(*) FROM admin_users")
        admin_count = cur.fetchone()[0]
        if admin_count == 0:
            import hashlib
            default_password = hashlib.sha256("admin123".encode()).hexdigest()
            cur.execute(
                "INSERT INTO admin_users (username, email, password, is_super_admin) VALUES (%s, %s, %s, %s) ON CONFLICT (email) DO NOTHING",
                ("admin", "admin@esticore.com", default_password, True)
            )
            print("✅ Default admin user created: admin@esticore.com / admin123")
        
        # 11. Insert default AI models (PostgreSQL)
        print("Creating default AI models...")
        default_models = [
            {
                "name": "GPT-4",
                "provider": "OpenAI",
                "model_name": "gpt-4",
                "is_active": True,
                "config": '{"temperature": 0.7, "max_tokens": 2000}'
            },
            {
                "name": "GPT-3.5-Turbo",
                "provider": "OpenAI", 
                "model_name": "gpt-3.5-turbo",
                "is_active": False,
                "config": '{"temperature": 0.7, "max_tokens": 2000}'
            },
            {
                "name": "Claude-2",
                "provider": "Anthropic",
                "model_name": "claude-2",
                "is_active": False,
                "config": '{"temperature": 0.7, "max_tokens": 2000}'
            }
        ]
        for model in default_models:
            cur.execute(
                "INSERT INTO ai_models (name, provider, model_name, is_active, config) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (name) DO NOTHING",
                (model["name"], model["provider"], model["model_name"], model["is_active"], model["config"])
            )
        
        conn.commit()
        print("✅ Database migration completed successfully!")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Migration failed: {str(e)}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()