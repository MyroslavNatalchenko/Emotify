'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Music, Upload, User } from 'lucide-react';
import Link from 'next/link';

export default function Home() {
  return (
    <div className="mx-auto max-w-md space-y-6 p-6">
      <header className="py-4 text-center">
        <h2 className="text-3xl font-bold text-balance text-slate-800">
          Odkryj emocje ukryte w Twojej muzyce
        </h2>
      </header>

      <div className="grid grid-cols-1 gap-4">
        <Link href="/spotify">
          <Card className="border-none bg-emerald-500 text-white shadow-lg transition-transform active:scale-95">
            <CardContent className="flex items-center gap-4 p-6">
              <div className="rounded-full bg-white/20 p-3">
                <Music size={24} />
              </div>
              <div>
                <h3 className="font-bold">Analiza Spotify</h3>
                <p className="text-xs opacity-80">Analizuj ostatnio słuchane utwory</p>
              </div>
            </CardContent>
          </Card>
        </Link>

        <Link href="/upload">
          <Card className="border-none bg-sky-500 text-white shadow-lg transition-transform active:scale-95">
            <CardContent className="flex items-center gap-4 p-6">
              <div className="rounded-full bg-white/20 p-3">
                <Upload size={24} />
              </div>
              <div>
                <h3 className="font-bold">Prześlij plik</h3>
                <p className="text-xs opacity-80">Analiza pojedynczego utworu</p>
              </div>
            </CardContent>
          </Card>
        </Link>

        <Link href="/account">
          <Card className="border-none bg-slate-800 text-white shadow-lg transition-transform active:scale-95">
            <CardContent className="flex items-center gap-4 p-6">
              <div className="rounded-full bg-white/20 p-3">
                <User size={24} />
              </div>
              <div>
                <h3 className="font-bold">Moje konto</h3>
                <p className="text-xs opacity-80">Zarządzaj połączeniem ze Spotify</p>
              </div>
            </CardContent>
          </Card>
        </Link>
      </div>
    </div>
  );
}
