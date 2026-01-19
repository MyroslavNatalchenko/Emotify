'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { getLoginUrl } from '@/lib/api';
import { User, Music, ChevronLeft, LogOut, ExternalLink, Loader2 } from 'lucide-react';
import Link from 'next/link';
import { cn } from '@/lib/utils';

export default function AccountPage() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('spotify_token');
    setIsLoggedIn(!!token);
    setIsLoading(false);
  }, []);

  const handleConnectSpotify = async () => {
    try {
      const targetPath = isLoggedIn ? '/spotify' : '/account';
      localStorage.setItem('auth_redirect', targetPath);

      const url = await getLoginUrl();
      window.location.href = url;
    } catch (error) {
      alert('Failed to get login URL.');
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('spotify_token');
    setIsLoggedIn(false);
  };

  return (
    <div className="mx-auto max-w-md space-y-6 p-6">
      <Link
        href="/"
        className="flex items-center gap-2 text-sm text-slate-500 transition-colors hover:text-emerald-500"
      >
        <ChevronLeft size={16} /> Powrót
      </Link>

      <header>
        <h2 className="text-2xl font-bold text-slate-800">Twoje Konto</h2>
        <p className="text-sm text-slate-500">Zarządzaj połączeniem ze Spotify</p>
      </header>

      <Card className="border-none bg-white shadow-xl shadow-slate-200/50">
        <CardContent className="space-y-8 px-6 pt-6 pb-2">
          <div className="flex items-center gap-4">
            <div
              className={cn(
                'flex h-16 w-16 items-center justify-center rounded-full border transition-colors',
                isLoggedIn
                  ? 'border-emerald-100 bg-emerald-50 text-emerald-500'
                  : 'border-slate-100 bg-slate-50 text-slate-400'
              )}
            >
              <User size={32} />
            </div>
            <div>
              <h3 className="text-lg font-bold text-slate-800">Użytkownik</h3>
              {isLoading ? (
                <div className="flex items-center gap-2 text-xs text-slate-400">
                  <Loader2 className="h-3 w-3 animate-spin" /> Sprawdzanie...
                </div>
              ) : isLoggedIn ? (
                <p className="inline-block rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-600">
                  Sesja aktywna
                </p>
              ) : (
                <p className="inline-block rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-500">
                  Niepołączony
                </p>
              )}
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="text-[10px] font-black tracking-[0.2em] text-slate-400 uppercase">
              Usługi zewnętrzne
            </h4>
            <div className="flex items-center justify-between rounded-2xl border border-slate-100 bg-slate-50 p-4 transition-all hover:bg-slate-100/50">
              <div className="flex items-center gap-3">
                <div
                  className={cn(
                    'rounded-xl bg-white p-2 shadow-sm',
                    isLoggedIn ? 'text-emerald-500' : 'text-slate-400'
                  )}
                >
                  <Music size={20} />
                </div>
                <div className="flex flex-col">
                  <span className="text-sm font-bold text-slate-700">Spotify API</span>
                  {!isLoggedIn && (
                    <span className="text-[10px] text-slate-400">
                      Wymagane do pobrania historii
                    </span>
                  )}
                </div>
              </div>

              <Button
                onClick={handleConnectSpotify}
                variant="ghost"
                size="sm"
                className="gap-2 text-xs font-bold text-emerald-600 hover:bg-emerald-100/50"
              >
                {isLoggedIn ? 'Odśwież token' : 'Połącz'} <ExternalLink size={14} />
              </Button>
            </div>
          </div>

          {isLoggedIn && (
            <div className="animate-in fade-in slide-in-from-top-2">
              <Button
                onClick={handleLogout}
                variant="destructive"
                className="w-full gap-2 border-none bg-red-50 font-bold text-red-600 shadow-none hover:bg-red-100"
              >
                <LogOut size={18} /> Wyloguj się
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
