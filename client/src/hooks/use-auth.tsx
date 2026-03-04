import { createContext, useContext, ReactNode } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiRequest } from '@/lib/queryClient';
import { getQueryFn } from '@/lib/queryClient';

interface User {
  id: string;
  username: string;
  email: string | null;
  age: number | null;
  deviceType: string | null;
  createdAt: string;
}

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  login: (credentials: LoginData) => Promise<void>;
  register: (credentials: RegisterData) => Promise<void>;
  logout: () => Promise<void>;
}

interface LoginData {
  username: string;
  password: string;
}

interface RegisterData {
  username: string;
  password: string;
  email?: string;
  age?: number;
  deviceType?: string;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();

  const {
    data: user,
    isLoading,
  } = useQuery<User | null>({
    queryKey: ['/api/auth/me'],
    queryFn: getQueryFn<User | null>({ on401: 'returnNull' }),
    staleTime: Infinity,
    retry: false,
  });

  const loginMutation = useMutation({
    mutationFn: async (credentials: LoginData) => {
      const res = await apiRequest('POST', '/api/auth/login', credentials);
      return res.json();
    },
    onSuccess: (data) => {
      // Store JWT for native (Capacitor) environments where cookies don't cross origins
      if (data.token) {
        try { localStorage.setItem('auth_token', data.token); } catch { /* private browsing */ }
      }
      queryClient.setQueryData(['/api/auth/me'], data.user);
    },
  });

  const registerMutation = useMutation({
    mutationFn: async (credentials: RegisterData) => {
      const res = await apiRequest('POST', '/api/auth/register', credentials);
      return res.json();
    },
    onSuccess: (data) => {
      if (data.token) {
        try { localStorage.setItem('auth_token', data.token); } catch { /* private browsing */ }
      }
      queryClient.setQueryData(['/api/auth/me'], data.user);
    },
  });

  const logoutMutation = useMutation({
    mutationFn: async () => {
      await apiRequest('POST', '/api/auth/logout');
    },
    onSuccess: () => {
      try { localStorage.removeItem('auth_token'); } catch { /* ok */ }
      queryClient.setQueryData(['/api/auth/me'], null);
      queryClient.clear();
    },
  });

  const login = async (credentials: LoginData) => {
    await loginMutation.mutateAsync(credentials);
  };

  const register = async (credentials: RegisterData) => {
    await registerMutation.mutateAsync(credentials);
  };

  const logout = async () => {
    await logoutMutation.mutateAsync();
  };

  return (
    <AuthContext.Provider
      value={{
        user: user ?? null,
        isLoading,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
