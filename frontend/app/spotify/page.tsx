'use client';

import { useState, useEffect } from 'react';
import { useMutation } from '@tanstack/react-query';
import { analyzeSpotify } from '@/lib/api';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { EmotionChart } from '@/components/emotion-chart';
import { Music, Play, Loader2, ChevronLeft, AlertCircle } from 'lucide-react';
import Link from 'next/link';

export default function SpotifyPage() {
  const [tracks, setTracks] = useState<any[]>([]);
  const [results, setResults] = useState<any | null>(null);
  const [analyzingId, setAnalyzingId] = useState<string | null>(null);

  const { mutate: analyze } = useMutation({
    mutationFn: ({ id, url }: { id: string; url: string }) => analyzeSpotify(id, url),
    onSuccess: (data) => {
      setResults(data);
      setAnalyzingId(null);
    },
    onError: () => {
      alert('Błąd analizy utworu.');
      setAnalyzingId(null);
    },
  });

  useEffect(() => {
    const token = localStorage.getItem('spotify_token');
    if (token) {
      fetch('https://api.spotify.com/v1/me/player/recently-played?limit=10', {
        headers: { Authorization: `Bearer ${token}` },
      })
        .then((res) => res.json())
        .then((data) => setTracks(data.items || []))
        .catch(console.error);
    }
  }, []);

  const handleAnalyze = (track: any) => {
    if (!track.preview_url) {
      alert('Ten utwór nie ma dostępnego preview do analizy.');
      return;
    }
    setAnalyzingId(track.id);
    analyze({ id: track.id, url: track.preview_url });
  };

  return (
    <div className="mx-auto max-w-md space-y-6 p-6">
      <Link
        href="/"
        className="flex items-center gap-2 text-sm text-slate-500 hover:text-emerald-500"
      >
        <ChevronLeft size={16} /> Powrót
      </Link>

      <header>
        <h2 className="text-2xl font-bold text-slate-800">Ostatnio słuchane</h2>
        <p className="text-sm text-slate-500">Wybierz utwór ze Spotify do analizy emocji</p>
      </header>

      <div className="space-y-3">
        {tracks.map((item: any, index: number) => (
          <Card
            key={`${item.track.id}-${index}`}
            className="overflow-hidden border-none bg-white shadow-sm"
          >
            <CardContent className="flex items-center justify-between p-4">
              <div className="flex items-center gap-3 overflow-hidden">
                <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded bg-slate-100">
                  <Music size={20} className="text-slate-400" />
                </div>
                <div className="truncate">
                  <p className="truncate text-sm font-bold text-slate-800">{item.track.name}</p>
                  <p className="truncate text-[10px] text-slate-400">
                    {item.track.artists[0].name}
                  </p>
                </div>
              </div>
              <Button
                onClick={() => handleAnalyze(item.track)}
                disabled={analyzingId === item.track.id}
                size="sm"
                className="h-8 bg-emerald-500 text-white hover:bg-emerald-600"
              >
                {analyzingId === item.track.id ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Play size={14} />
                )}
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>

      {results && (
        <div className="animate-in fade-in slide-in-from-bottom-4 pt-4">
          <Card className="overflow-hidden border-none bg-white shadow-xl">
            <div className="flex items-center justify-between border-b border-emerald-100 bg-emerald-50 p-4">
              <h3 className="text-sm font-bold text-emerald-800">Wynik analizy Spotify</h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setResults(null)}
                className="h-6 px-2 text-emerald-600"
              >
                Zamknij
              </Button>
            </div>
            <CardContent className="p-6">
              <EmotionChart data={results} />
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
