'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { getLoginUrl } from '@/lib/api';
import { User, Music, ChevronLeft, ExternalLink } from 'lucide-react';
import Link from 'next/link';

export default function AccountPage() {
  const handleConnectSpotify = async () => {
    try {
      const url = await getLoginUrl();
      window.location.href = url;
    } catch (error) {
      alert('Nie udało się pobrać linku do logowania.');
    }
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
      </header>

      <Card className="border-none bg-white shadow-lg">
        <CardContent className="space-y-6 p-6">
          <div className="flex items-center gap-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-slate-100 text-slate-400">
              <User size={32} />
            </div>
            <div>
              <h3 className="font-bold text-slate-800">Użytkownik</h3>
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="text-xs font-bold tracking-widest text-slate-400 uppercase">
              Usługi zewnętrzne
            </h4>
            <div className="flex items-center justify-between rounded-xl border border-slate-100 p-4">
              <div className="flex items-center gap-3">
                <div className="text-emerald-500">
                  <Music size={20} />
                </div>
                <span className="text-sm font-medium">Spotify</span>
              </div>
              <Button
                onClick={handleConnectSpotify}
                variant="outline"
                size="sm"
                className="gap-2 border-emerald-200 text-emerald-600 hover:bg-emerald-50"
              >
                Połącz <ExternalLink size={14} />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
