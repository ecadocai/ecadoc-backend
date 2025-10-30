"""
Script to initialize subscription plans in the database
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.database import db_manager

def initialize_subscription_plans():
    """Initialize the subscription plans"""
    
    plans = [
        {
            "name": "Solo Plan",
            "description": "Project Repository & Document Upload (up to 3 active projects, 15GB storage)",
            "six_month_price": 0.0,  # Free trial
            "annual_price": 99.0,
            "storage_gb": 15,
            "project_limit": 3,
            "user_limit": 1,
            "action_limit": 350,
            "features": [
                "Version Control for document updates",
                "Natural Language AI Chat (multi-turn chat, source citations)",
                "Embedded PDF Viewer",
                "Manual annotation Tools",
                "Limited AI-Driven Markup/measurements (350 action limit per month)",
                "Standard Email Support",
                "Full Bluebeam Studio Sync"
            ],
            "has_free_trial": True,
            "trial_days": 30,
            "currency": "usd"
        },
        {
            "name": "Team Plan",
            "description": "Team Collaboration (shared project folders, concurrent editing)",
            "six_month_price": 98.0,
            "annual_price": 499.0,
            "storage_gb": 50,
            "project_limit": 10,
            "user_limit": 5,
            "action_limit": 2000,
            "features": [
                "Everything in Solo Plan",
                "Team Collaboration (shared project folders, concurrent editing)",
                "Expanded Storage (50GB)",
                "Increased AI-Driven mark-ups/Measurement(2000 action limit per month)",
                "Bluebeam Studio Export (push AI-marked PDFs)",
                "Priority Email + Chat Support",
                "Project History Retention: up to 10 versions per project"
            ],
            "has_free_trial": False,
            "trial_days": 0,
            "currency": "usd"
        },
        {
            "name": "Business Plan",
            "description": "Advanced features for business teams",
            "six_month_price": 198.0,
            "annual_price": 999.0,
            "storage_gb": 100,
            "project_limit": 20,
            "user_limit": 20,
            "action_limit": 5000,
            "features": [
                "Everything in Team Plan",
                "Unlimited AI-driven mark-ups/measurement",
                "Role-Based Permissions & SSO",
                "Dedicated Onboarding & Training",
                "100GB Storage",
                "Project History Retention: up to 20 versions per project"
            ],
            "has_free_trial": False,
            "trial_days": 0,
            "currency": "usd"
        },
        {
            "name": "Enterprise Plan",
            "description": "Enterprise-grade features for large organizations",
            "six_month_price": 398.0,
            "annual_price": 1999.0,
            "storage_gb": 300,
            "project_limit": 50,
            "user_limit": 50,
            "action_limit": 10000,
            "features": [
                "Everything in Business Plan",
                "300GB Storage",
                "Project History Retention: up to 30 versions per project",
                "Custom AI Model Integration (fine-tuned per client data)",
                "Advanced Security Features (audit logs, encryption customization)",
                "Custom Integrations (ERP, CRM, proprietary project systems)",
                "Dedicated Account Manager",
                "Quarterly Success Reviews & Workflow Optimization Consulting"
            ],
            "has_free_trial": False,
            "trial_days": 0,
            "currency": "usd"
        }
    ]
    
    for plan_data in plans:
        try:
            # Check if plan already exists
            existing_plan = db_manager.get_subscription_plan_by_name(plan_data["name"])
            if not existing_plan:
                db_manager.create_subscription_plan(
                    name=plan_data["name"],
                    description=plan_data["description"],
                    storage_gb=plan_data["storage_gb"],
                    project_limit=plan_data["project_limit"],
                    user_limit=plan_data["user_limit"],
                    action_limit=plan_data["action_limit"],
                    features=plan_data["features"],
                    has_free_trial=plan_data["has_free_trial"],
                    trial_days=plan_data["trial_days"],
                    prices=[
                        {
                            "duration_months": 6,
                            "price": plan_data["six_month_price"],
                            "currency": plan_data.get("currency", "usd"),
                        },
                        {
                            "duration_months": 12,
                            "price": plan_data["annual_price"],
                            "currency": plan_data.get("currency", "usd"),
                        },
                    ],
                )
                print(f"✅ Created plan: {plan_data['name']}")
            else:
                print(f"⚠️ Plan already exists: {plan_data['name']}")
        except Exception as e:
            print(f"❌ Error creating plan {plan_data['name']}: {str(e)}")

if __name__ == "__main__":
    print("Initializing subscription plans...")
    initialize_subscription_plans()
    print("Subscription plans initialization completed!")