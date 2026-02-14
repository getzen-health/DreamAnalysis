import { useState } from 'react';
import { useLocation } from 'wouter';
import { useAuth } from '@/hooks/use-auth';
import { useToast } from '@/hooks/use-toast';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Brain, LogIn, UserPlus, Loader2 } from 'lucide-react';

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
  const [registerLoading, setRegisterLoading] = useState(false);

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
      setLocation('/');
    } catch (err: any) {
      const message = err.message?.includes(':')
        ? err.message.split(':').slice(1).join(':').trim()
        : 'Invalid username or password.';
      let parsed = message;
      try {
        const obj = JSON.parse(message);
        if (obj.error) parsed = obj.error;
      } catch {
        // not JSON, use as-is
      }
      toast({
        title: 'Login Failed',
        description: parsed,
        variant: 'destructive',
      });
    } finally {
      setLoginLoading(false);
    }
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!registerUsername.trim()) {
      toast({
        title: 'Validation Error',
        description: 'Username is required.',
        variant: 'destructive',
      });
      return;
    }

    if (registerUsername.trim().length < 3) {
      toast({
        title: 'Validation Error',
        description: 'Username must be at least 3 characters long.',
        variant: 'destructive',
      });
      return;
    }

    if (!registerPassword || registerPassword.length < 6) {
      toast({
        title: 'Validation Error',
        description: 'Password must be at least 6 characters long.',
        variant: 'destructive',
      });
      return;
    }

    if (registerPassword !== confirmPassword) {
      toast({
        title: 'Validation Error',
        description: 'Passwords do not match.',
        variant: 'destructive',
      });
      return;
    }

    setRegisterLoading(true);
    try {
      await register({
        username: registerUsername.trim(),
        password: registerPassword,
        email: registerEmail.trim() || undefined,
      });
      toast({
        title: 'Account Created',
        description: 'Welcome to Neural Dream Workshop!',
      });
      setLocation('/');
    } catch (err: any) {
      const message = err.message?.includes(':')
        ? err.message.split(':').slice(1).join(':').trim()
        : 'Registration failed. Please try again.';
      let parsed = message;
      try {
        const obj = JSON.parse(message);
        if (obj.error) parsed = obj.error;
      } catch {
        // not JSON, use as-is
      }
      toast({
        title: 'Registration Failed',
        description: parsed,
        variant: 'destructive',
      });
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
          <div className="flex items-center justify-center space-x-3 mb-6">
            <div className="relative">
              <Brain className="h-16 w-16 text-primary" />
              <div className="absolute inset-0 h-16 w-16 bg-primary/20 rounded-full blur-xl animate-pulse" />
            </div>
          </div>
          <h1 className="text-5xl font-futuristic font-bold text-gradient leading-tight">
            Neural Dream Workshop
          </h1>
          <p className="text-xl text-foreground/70 leading-relaxed">
            Unlock the secrets of your mind through advanced neural monitoring,
            AI-powered dream analysis, and real-time brain-computer interface technology.
          </p>
          <div className="grid grid-cols-3 gap-6 pt-6">
            <div className="glass-card rounded-xl p-4 border border-primary/20">
              <div className="text-2xl font-bold text-primary font-mono">64</div>
              <div className="text-xs text-foreground/60 mt-1">EEG Channels</div>
            </div>
            <div className="glass-card rounded-xl p-4 border border-secondary/20">
              <div className="text-2xl font-bold text-secondary font-mono">AI</div>
              <div className="text-xs text-foreground/60 mt-1">Dream Analysis</div>
            </div>
            <div className="glass-card rounded-xl p-4 border border-accent/20">
              <div className="text-2xl font-bold text-accent font-mono">24/7</div>
              <div className="text-xs text-foreground/60 mt-1">Monitoring</div>
            </div>
          </div>
        </div>
      </div>

      {/* Right panel: Auth forms */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-6 relative z-10">
        <Card className="w-full max-w-md glass-card border border-primary/20 shadow-2xl shadow-primary/5">
          <CardHeader className="text-center pb-2">
            <div className="flex items-center justify-center space-x-2 mb-4 lg:hidden">
              <Brain className="h-8 w-8 text-primary" />
              <span className="text-xl font-futuristic font-bold text-gradient">Neural Dream Workshop</span>
            </div>
            <CardTitle className="text-2xl font-futuristic text-gradient">
              Access Portal
            </CardTitle>
            <CardDescription className="text-foreground/60">
              Enter your neural workspace
            </CardDescription>
          </CardHeader>

          <CardContent>
            <Tabs defaultValue="login" className="w-full">
              <TabsList className="grid w-full grid-cols-2 bg-card/50 border border-primary/10">
                <TabsTrigger
                  value="login"
                  className="data-[state=active]:bg-primary/20 data-[state=active]:text-primary font-medium"
                >
                  <LogIn className="h-4 w-4 mr-2" />
                  Login
                </TabsTrigger>
                <TabsTrigger
                  value="register"
                  className="data-[state=active]:bg-primary/20 data-[state=active]:text-primary font-medium"
                >
                  <UserPlus className="h-4 w-4 mr-2" />
                  Register
                </TabsTrigger>
              </TabsList>

              {/* Login Tab */}
              <TabsContent value="login" className="mt-6">
                <form onSubmit={handleLogin} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="login-username" className="text-foreground/80">
                      Username
                    </Label>
                    <Input
                      id="login-username"
                      type="text"
                      placeholder="Enter your username"
                      value={loginUsername}
                      onChange={(e) => setLoginUsername(e.target.value)}
                      className="bg-card/50 border-primary/20 focus:border-primary/50 focus:ring-primary/20"
                      autoComplete="username"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="login-password" className="text-foreground/80">
                      Password
                    </Label>
                    <Input
                      id="login-password"
                      type="password"
                      placeholder="Enter your password"
                      value={loginPassword}
                      onChange={(e) => setLoginPassword(e.target.value)}
                      className="bg-card/50 border-primary/20 focus:border-primary/50 focus:ring-primary/20"
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
                </form>
              </TabsContent>

              {/* Register Tab */}
              <TabsContent value="register" className="mt-6">
                <form onSubmit={handleRegister} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="register-username" className="text-foreground/80">
                      Username
                    </Label>
                    <Input
                      id="register-username"
                      type="text"
                      placeholder="Choose a username (3+ characters)"
                      value={registerUsername}
                      onChange={(e) => setRegisterUsername(e.target.value)}
                      className="bg-card/50 border-primary/20 focus:border-primary/50 focus:ring-primary/20"
                      autoComplete="username"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="register-email" className="text-foreground/80">
                      Email <span className="text-foreground/40">(optional)</span>
                    </Label>
                    <Input
                      id="register-email"
                      type="email"
                      placeholder="your@email.com"
                      value={registerEmail}
                      onChange={(e) => setRegisterEmail(e.target.value)}
                      className="bg-card/50 border-primary/20 focus:border-primary/50 focus:ring-primary/20"
                      autoComplete="email"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="register-password" className="text-foreground/80">
                      Password
                    </Label>
                    <Input
                      id="register-password"
                      type="password"
                      placeholder="Create a password (6+ characters)"
                      value={registerPassword}
                      onChange={(e) => setRegisterPassword(e.target.value)}
                      className="bg-card/50 border-primary/20 focus:border-primary/50 focus:ring-primary/20"
                      autoComplete="new-password"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="confirm-password" className="text-foreground/80">
                      Confirm Password
                    </Label>
                    <Input
                      id="confirm-password"
                      type="password"
                      placeholder="Confirm your password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      className="bg-card/50 border-primary/20 focus:border-primary/50 focus:ring-primary/20"
                      autoComplete="new-password"
                    />
                  </div>
                  <Button
                    type="submit"
                    disabled={registerLoading}
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
