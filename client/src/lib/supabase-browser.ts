let _supabase: any = null

export async function getSupabase() {
  if (_supabase) return _supabase

  const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
  const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY

  if (!supabaseUrl || !supabaseKey) {
    console.warn('[supabase-browser] VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY not set')
    return null
  }

  const { createClient } = await import('@supabase/supabase-js')
  _supabase = createClient(supabaseUrl, supabaseKey)
  return _supabase
}
