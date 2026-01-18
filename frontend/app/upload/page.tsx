'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { analyzeFile } from '@/lib/api';
import { Card, CardContent } from '@/components/ui/card';
import { EmotionChart } from '@/components/emotion-chart';
import { Upload, Loader2, Music, ChevronLeft } from 'lucide-react';
import Link from 'next/link';
import { cn } from '@/lib/utils';

interface Emotion {
  tag: string;
  confidence: number;
}

export default function UploadPage() {
  const [results, setResults] = useState<Emotion[] | null>(null);

  const { mutate, isPending } = useMutation({
    mutationFn: analyzeFile,
    onSuccess: (data) => setResults(data),
    onError: () => alert('Błąd analizy pliku.'),
  });

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) mutate(file);
  };

  return (
    <div className="mx-auto max-w-md space-y-6 p-6">
      <Link
        href="/"
        className="flex items-center gap-2 text-sm text-slate-500 transition-colors hover:text-emerald-500"
      >
        <ChevronLeft size={16} /> Powrót do menu
      </Link>

      <header>
        <h2 className="text-2xl font-bold text-slate-800">Analiza pliku MP3</h2>
        <p className="text-sm text-slate-500">
          Wybierz utwór z dysku, aby poznać jego profil emocjonalny
        </p>
      </header>

      <Card className="border-2 border-dashed border-slate-200 bg-white shadow-none transition-colors hover:border-sky-400">
        <CardContent className="p-8 text-center">
          <input
            type="file"
            id="manual-upload"
            className="hidden"
            accept=".mp3,.wav"
            onChange={handleFile}
            disabled={isPending}
          />
          <label
            htmlFor="manual-upload"
            className={cn(
              'flex cursor-pointer flex-col items-center gap-4',
              isPending && 'cursor-not-allowed opacity-50'
            )}
          >
            <div className="rounded-full bg-sky-50 p-4 text-sky-500">
              {isPending ? <Loader2 className="h-8 w-8 animate-spin" /> : <Upload size={32} />}
            </div>
            <div className="space-y-1">
              <p className="font-bold text-slate-700">
                {isPending ? 'Analizowanie utworu...' : 'Kliknij, aby przesłać'}
              </p>
              <p className="text-xs text-slate-400">MP3 lub WAV (max 16MB)</p>
            </div>
          </label>
        </CardContent>
      </Card>

      {results && (
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
          <Card className="overflow-hidden border-none bg-white shadow-xl">
            <div className="border-b border-sky-100 bg-sky-50 p-4">
              <h3 className="flex items-center gap-2 text-sm font-bold text-sky-800">
                <Music size={16} /> Wynik analizy
              </h3>
            </div>
            <CardContent className="p-6">
              <div className="mb-6">
                <span className="mb-2 inline-block rounded-full bg-sky-100 px-3 py-1 text-xs font-bold tracking-wider text-sky-700 uppercase">
                  Dominująca: {results[0]?.tag}
                </span>
                <EmotionChart data={results} />
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
