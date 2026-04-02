import { useState, useMemo } from 'react';
import { useLocation } from 'wouter';
import { apiRequest } from '@/lib/queryClient';
import { useAuth } from '@/hooks/use-auth';
import { useToast } from '@/hooks/use-toast';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { LogIn, UserPlus, Loader2, Check, X as XIcon } from 'lucide-react';
import { sbGetSetting } from "../lib/supabase-store";

// ── Password & username validation ───────────────────────────────────────────

interface PasswordCheck {
  label: string;
  met: boolean;
}

function getPasswordChecks(password: string): PasswordCheck[] {
  return [
    { label: "At least 8 characters", met: password.length >= 8 },
    { label: "One uppercase letter", met: /[A-Z]/.test(password) },
    { label: "One lowercase letter", met: /[a-z]/.test(password) },
    { label: "One number", met: /[0-9]/.test(password) },
    { label: "One special character (!@#$%^&*()_+-=)", met: /[!@#$%^&*()_+\-=]/.test(password) },
  ];
}

function getPasswordStrength(checks: PasswordCheck[]): { score: number; label: string; color: string } {
  const met = checks.filter((c) => c.met).length;
  if (met <= 1) return { score: met, label: "Very weak", color: "bg-rose-500" };
  if (met === 2) return { score: met, label: "Weak", color: "bg-orange-500" };
  if (met === 3) return { score: met, label: "Fair", color: "bg-amber-500" };
  if (met === 4) return { score: met, label: "Good", color: "bg-indigo-500" };
  return { score: met, label: "Strong", color: "bg-cyan-600" };
}

function getUsernameError(username: string): string | null {
  if (!username) return null; // don't show error on empty
  if (username.length < 3) return "Username must be at least 3 characters.";
  if (!/^[a-zA-Z0-9_]+$/.test(username)) return "Only letters, numbers, and underscores allowed.";
  return null;
}

export default function AuthPage() {
  const [, setLocation] = useLocation();
  const { user, login, register } = useAuth();
  const { toast } = useToast();

  // Login form state
  const [loginUsername, setLoginUsername] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginLoading, setLoginLoading] = useState(false);

  // Register form state
  const [registerUsername, setRegisterUsername] = useState('');
  const [registerEmail, setRegisterEmail] = useState('');
  const [registerPassword, setRegisterPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [registerAge, setRegisterAge] = useState('');
  const [registerLoading, setRegisterLoading] = useState(false);

  // After login/register, route based on stored intent
  async function redirectByIntent() {
    try {
      const res = await apiRequest('GET', '/api/user/intent');
      const data = await res.json();
      if (data.intent === 'study') {
        setLocation('/study');
      } else if (data.intent === 'explore') {
        setLocation('/');
      } else {
        setLocation('/intent'); // first-time user — pick intent
      }
    } catch {
      setLocation('/intent');
    }
  }

  // Redirect if already authenticated
  if (user) {
    setLocation('/');
    return null;
  }

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!loginUsername.trim() || !loginPassword) {
      toast({
        title: 'Validation Error',
        description: 'Please fill in all fields.',
        variant: 'destructive',
      });
      return;
    }

    setLoginLoading(true);
    try {
      await login({ username: loginUsername.trim(), password: loginPassword });
      toast({
        title: 'Welcome back',
        description: 'You have been logged in successfully.',
      });
      await redirectByIntent();
    } catch (err: any) {
      const status = err.status as number | undefined;
      const description =
        status === 401 ? 'Wrong username or password.' :
        status === 429 ? 'Too many attempts. Please wait a moment.' :
        status === 500 ? 'Something went wrong on our end. Please try again.' :
        !status ? 'Cannot reach the server. Check your connection.' :
        err.message || 'Login failed. Please try again.';
      toast({ title: 'Login Failed', description, variant: 'destructive' });
    } finally {
      setLoginLoading(false);
    }
  };

  // Inline validation state
  const passwordChecks = useMemo(() => getPasswordChecks(registerPassword), [registerPassword]);
  const passwordStrength = useMemo(() => getPasswordStrength(passwordChecks), [passwordChecks]);
  const allPasswordChecksMet = passwordChecks.every((c) => c.met);
  const usernameError = getUsernameError(registerUsername.trim());
  const passwordsMatch = registerPassword === confirmPassword;
  const confirmTouched = confirmPassword.length > 0;

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();

    // Client-side gates — inline errors handle display
    if (!registerUsername.trim() || usernameError) return;
    if (!allPasswordChecksMet) return;
    if (!passwordsMatch) return;

    setRegisterLoading(true);
    try {
      await register({
        username: registerUsername.trim(),
        password: registerPassword,
        email: registerEmail.trim() || undefined,
        age: registerAge ? Number(registerAge) : undefined,
      });
      toast({
        title: 'Account Created',
        description: 'Welcome to AntarAI! Your emotional health journey starts now.',
      });
      // New users see welcome intro; returning users go through intent flow
      const seen = sbGetSetting("onboarding_complete");
      if (!seen) {
        setLocation("/welcome-intro");
      } else {
        await redirectByIntent();
      }
    } catch (err: any) {
      const status = err.status as number | undefined;
      const description =
        status === 409 ? 'That username is already taken. Please choose another.' :
        status === 400 ? (err.message || 'Please check your details and try again.') :
        status === 500 ? 'Something went wrong on our end. Please try again in a moment.' :
        !status ? 'Cannot reach the server. Check your connection.' :
        err.message || 'Registration failed. Please try again.';
      toast({ title: 'Registration Failed', description, variant: 'destructive' });
    } finally {
      setRegisterLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground relative overflow-hidden flex">
      {/* Animated neural background */}
      <div className="neural-bg" />

      {/* Floating gradient orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-pulse pointer-events-none" />
      <div
        className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-secondary/10 rounded-full blur-3xl animate-pulse pointer-events-none"
        style={{ animationDelay: '1s' }}
      />
      <div
        className="absolute top-1/2 left-1/2 w-64 h-64 bg-accent/5 rounded-full blur-3xl animate-pulse pointer-events-none"
        style={{ animationDelay: '2s' }}
      />

      {/* Left panel: Hero / Branding */}
      <div className="hidden lg:flex lg:w-1/2 items-center justify-center p-12 relative z-10">
        <div className="max-w-lg text-center space-y-8">
          <div className="flex items-center justify-center mb-6">
            <div className="relative">
              <img src="/logo-antarai.svg" alt="AntarAI" className="h-20 w-20" />
              <div className="absolute inset-0 h-20 w-20 bg-primary/20 rounded-full blur-xl animate-pulse" />
            </div>
          </div>
          <h1 className="text-5xl font-futuristic font-bold text-gradient leading-tight">
            AntarAI
          </h1>
          <p className="text-xl text-foreground/70 leading-relaxed">
            Your emotional health companion. Track mood, stress, and focus
            through voice analysis, EEG, and real-time biometrics.
          </p>
          <div className="grid grid-cols-3 gap-6 pt-6">
            <div className="glass-card rounded-xl p-4 border border-primary/20">
              <div className="text-2xl font-bold text-primary font-mono">Voice</div>
              <div className="text-xs text-foreground/60 mt-1">Emotion Analysis</div>
            </div>
            <div className="glass-card rounded-xl p-4 border border-secondary/20">
              <div className="text-2xl font-bold text-secondary font-mono">EEG</div>
              <div className="text-xs text-foreground/60 mt-1">Brain Insights</div>
            </div>
            <div className="glass-card rounded-xl p-4 border border-accent/20">
              <div className="text-2xl font-bold text-accent font-mono">Health</div>
              <div className="text-xs text-foreground/60 mt-1">Daily Tracking</div>
            </div>
          </div>
        </div>
      </div>

      {/* Right panel: Auth forms */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-6 relative z-10">
        <Card className="w-full max-w-md glass-card border border-primary/20 shadow-2xl shadow-primary/5">
          <CardHeader className="text-center pb-2">
            <div className="flex items-center justify-center space-x-2 mb-4 lg:hidden">
              <img src="/logo-antarai.svg" alt="AntarAI" className="h-10 w-10" />
              <span className="text-xl font-futuristic font-bold text-gradient">AntarAI</span>
            </div>
            <CardTitle className="text-2xl font-futuristic text-gradient">
              Access Portal
            </CardTitle>
            <CardDescription className="text-foreground/60">
              Sign in to continue
            </CardDescription>
          </CardHeader>

          <CardContent>
            <Tabs defaultValue="login" className="w-full">
              <TabsList className="grid w-full grid-cols-2 border border-primary/10" style={{ backgroundColor: 'var(--muted)' }}>
                <TabsTrigger
                  value="login"
                  className="data-[state=active]:bg-primary/20 data-[state=active]:text-primary font-medium"
                  style={{ color: 'var(--muted-foreground)' }}
                >
                  <LogIn className="h-4 w-4 mr-2" />
                  Login
                </TabsTrigger>
                <TabsTrigger
                  value="register"
                  className="data-[state=active]:bg-primary/20 data-[state=active]:text-primary font-medium"
                  style={{ color: 'var(--muted-foreground)' }}
                >
                  <UserPlus className="h-4 w-4 mr-2" />
                  Register
                </TabsTrigger>
              </TabsList>

              {/* Login Tab */}
              <TabsContent value="login" className="mt-6">
                <form onSubmit={handleLogin} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="login-username" className="text-foreground">
                      Username
                    </Label>
                    <Input
                      id="login-username"
                      type="text"
                      placeholder="Enter your username"
                      value={loginUsername}
                      onChange={(e) => setLoginUsername(e.target.value)}
                      className="border-primary/20 focus:border-primary/50 focus:ring-primary/20"
                      style={{ color: 'var(--foreground)', backgroundColor: 'var(--input)' }}
                      autoComplete="username"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="login-password" className="text-foreground">
                      Password
                    </Label>
                    <Input
                      id="login-password"
                      type="password"
                      placeholder="Enter your password"
                      value={loginPassword}
                      onChange={(e) => setLoginPassword(e.target.value)}
                      className="border-primary/20 focus:border-primary/50 focus:ring-primary/20"
                      style={{ color: 'var(--foreground)', backgroundColor: 'var(--input)' }}
                      autoComplete="current-password"
                    />
                  </div>
                  <Button
                    type="submit"
                    disabled={loginLoading}
                    className="w-full bg-primary/90 hover:bg-primary text-primary-foreground font-medium transition-all duration-300 hover:shadow-lg hover:shadow-primary/25"
                  >
                    {loginLoading ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Connecting...
                      </>
                    ) : (
                      <>
                        <LogIn className="h-4 w-4 mr-2" />
                        Sign In
                      </>
                    )}
                  </Button>
                  <div className="text-center">
                    <a
                      href="/forgot-password"
                      className="text-sm text-muted-foreground hover:text-primary transition-colors"
                    >
                      Forgot password?
                    </a>
                  </div>
                </form>
              </TabsContent>

              {/* Register Tab */}
              <TabsContent value="register" className="mt-6">
                <form onSubmit={handleRegister} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="register-username" className="text-foreground">
                      Username
                    </Label>
                    <Input
                      id="register-username"
                      type="text"
                      placeholder="Choose a username (3+ characters)"
                      value={registerUsername}
                      onChange={(e) => setRegisterUsername(e.target.value)}
                      className="border-primary/20 focus:border-primary/50 focus:ring-primary/20"
                      style={{ color: 'var(--foreground)', backgroundColor: 'var(--input)' }}
                      autoComplete="username"
                    />
                    {usernameError && (
                      <p className="text-xs text-rose-400">{usernameError}</p>
                    )}
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="register-email" className="text-foreground">
                      Email <span className="text-foreground/40">(optional)</span>
                    </Label>
                    <Input
                      id="register-email"
                      type="email"
                      placeholder="your@email.com"
                      value={registerEmail}
                      onChange={(e) => setRegisterEmail(e.target.value)}
                      className="border-primary/20 focus:border-primary/50 focus:ring-primary/20"
                      style={{ color: 'var(--foreground)', backgroundColor: 'var(--input)' }}
                      autoComplete="email"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="register-password" className="text-foreground">
                      Password
                    </Label>
                    <Input
                      id="register-password"
                      type="password"
                      placeholder="Create a strong password"
                      value={registerPassword}
                      onChange={(e) => setRegisterPassword(e.target.value)}
                      className="border-primary/20 focus:border-primary/50 focus:ring-primary/20"
                      style={{ color: 'var(--foreground)', backgroundColor: 'var(--input)' }}
                      autoComplete="new-password"
                    />
                    {registerPassword.length > 0 && (
                      <div className="space-y-2">
                        {/* Strength bar */}
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-1.5 rounded-full bg-muted/40 overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-300 ${passwordStrength.color}`}
                              style={{ width: `${(passwordStrength.score / 5) * 100}%` }}
                            />
                          </div>
                          <span className={`text-xs font-medium ${passwordStrength.color.replace('bg-', 'text-')}`}>
                            {passwordStrength.label}
                          </span>
                        </div>
                        {/* Requirement checklist */}
                        <ul className="space-y-0.5">
                          {passwordChecks.map((check) => (
                            <li key={check.label} className="flex items-center gap-1.5 text-xs">
                              {check.met ? (
                                <Check className="h-3 w-3 text-cyan-400 shrink-0" />
                              ) : (
                                <XIcon className="h-3 w-3 text-muted-foreground/50 shrink-0" />
                              )}
                              <span className={check.met ? "text-cyan-400" : "text-muted-foreground/60"}>
                                {check.label}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="confirm-password" className="text-foreground">
                      Confirm Password
                    </Label>
                    <Input
                      id="confirm-password"
                      type="password"
                      placeholder="Confirm your password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      className="border-primary/20 focus:border-primary/50 focus:ring-primary/20"
                      style={{ color: 'var(--foreground)', backgroundColor: 'var(--input)' }}
                      autoComplete="new-password"
                    />
                    {confirmTouched && (
                      <p className={`text-xs flex items-center gap-1 ${passwordsMatch ? "text-cyan-400" : "text-rose-400"}`}>
                        {passwordsMatch ? (
                          <><Check className="h-3 w-3" /> Passwords match</>
                        ) : (
                          <><XIcon className="h-3 w-3" /> Passwords do not match</>
                        )}
                      </p>
                    )}
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-2">
                      <Label htmlFor="register-age" className="text-foreground">
                        Age <span className="text-foreground/40">(optional)</span>
                      </Label>
                      <Input
                        id="register-age"
                        type="number"
                        placeholder="e.g. 28"
                        min={10} max={120}
                        value={registerAge}
                        onChange={(e) => setRegisterAge(e.target.value)}
                        className="border-primary/20 focus:border-primary/50 focus:ring-primary/20"
                      style={{ color: 'var(--foreground)', backgroundColor: 'var(--input)' }}
                      />
                    </div>
                  </div>
                  <Button
                    type="submit"
                    disabled={registerLoading || !allPasswordChecksMet || !passwordsMatch || !!usernameError || !registerUsername.trim()}
                    className="w-full bg-primary/90 hover:bg-primary text-primary-foreground font-medium transition-all duration-300 hover:shadow-lg hover:shadow-primary/25"
                  >
                    {registerLoading ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Creating Account...
                      </>
                    ) : (
                      <>
                        <UserPlus className="h-4 w-4 mr-2" />
                        Create Account
                      </>
                    )}
                  </Button>
                </form>
              </TabsContent>
            </Tabs>

            {/* Neural decorative line */}
            <div className="mt-6 pt-4 border-t border-primary/10 text-center">
              <p className="text-xs text-foreground/40">
                Secured with end-to-end neural encryption
              </p>
              <div className="flex justify-center mt-3 space-x-1">
                {[0, 0.2, 0.4, 0.6, 0.8].map((delay, i) => (
                  <div
                    key={i}
                    className="w-1.5 h-1.5 rounded-full bg-primary/40 animate-pulse"
                    style={{ animationDelay: `${delay}s` }}
                  />
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
