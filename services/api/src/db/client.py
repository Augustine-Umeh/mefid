"""
supabase client instance for database interactions
"""

from supabase import acreate_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_ADMIN_API_KEY = os.getenv("SUPABASE_ADMIN_API_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# supabase client instance
supabase_client: Client = acreate_client(SUPABASE_DB_URL, SUPABASE_SERVICE_ROLE_KEY)
