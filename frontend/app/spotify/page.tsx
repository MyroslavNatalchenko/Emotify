'use client';

import { useState, useEffect, useCallback } from 'react';
import { useMutation } from '@tanstack/react-query';
import { analyzeSpotify, getLoginUrl } from '@/lib/api';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { EmotionChart } from '@/components/emotion-chart';
import { Music, Play, Loader2, ChevronLeft, RefreshCw, LogIn, ExternalLink } from 'lucide-react';
import Link from 'next/link';
import { cn } from '@/lib/utils';

export default function SpotifyPage() {
  const [tracks, setTracks] = useState<any[]>([]);
  const [results, setResults] = useState<any | null>(null);

  const [activeTrackId, setActiveTrackId] = useState<string | null>(null);
  const [analyzingId, setAnalyzingId] = useState<string | null>(null);

  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isLoadingAuth, setIsLoadingAuth] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('spotify_token');
    setIsLoggedIn(!!token);
    setIsLoadingAuth(false);
  }, []);

  const fetchTracks = useCallback(async () => {
    const token = localStorage.getItem('spotify_token');
    if (!token) return;

    setIsRefreshing(true);
    try {
      const res = await fetch(`https://api.spotify.com/v1/me/player/recently-played?limit=20`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.status === 401) {
        setIsLoggedIn(false);
        localStorage.removeItem('spotify_token');
        return;
      }

      const data = await res.json();
      setTracks(data.items || []);
    } catch (error) {
      console.error('Error fetching tracks:', error);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    if (isLoggedIn) {
      fetchTracks();
    }
  }, [isLoggedIn, fetchTracks]);

  const { mutate: analyze } = useMutation({
    mutationFn: (trackId: string) => analyzeSpotify(trackId),
    onSuccess: (data) => {
      setResults(data);
      setAnalyzingId(null);
    },
    onError: (error: any) => {
      const errorMsg = error.response?.data?.error || 'Track analysis error.';
      alert(errorMsg);
      setAnalyzingId(null);
      setActiveTrackId(null);
    },
  });

  const handleAnalyze = (track: any) => {
    if (activeTrackId === track.id) {
      setActiveTrackId(null);
      setResults(null);
      return;
    }

    setResults(null);
    setActiveTrackId(track.id);
    setAnalyzingId(track.id);
    analyze(track.id);
  };

  const handleConnect = async () => {
    try {
      localStorage.setItem('auth_redirect', '/spotify');
      const url = await getLoginUrl();
      window.location.href = url;
    } catch (e) {
      alert('Login error');
    }
  };

  if (isLoadingAuth) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-emerald-500" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-md space-y-6 p-6">
      <div className="flex items-center justify-between">
        <Link
          href="/"
          className="flex items-center gap-2 text-sm text-slate-500 hover:text-emerald-500"
        >
          <ChevronLeft size={16} /> Powrót
        </Link>

        {isLoggedIn && (
          <Button
            variant="ghost"
            size="sm"
            onClick={fetchTracks}
            disabled={isRefreshing}
            className="h-8 w-8 p-0 text-slate-400 hover:text-emerald-500"
          >
            <RefreshCw size={16} className={isRefreshing ? 'animate-spin' : ''} />
          </Button>
        )}
      </div>

      <header>
        <h2 className="text-2xl font-bold text-slate-800">Ostatnio słuchane</h2>
        <p className="text-sm text-slate-500">
          {isLoggedIn
            ? 'Wybierz utwór ze Spotify do analizy emocji'
            : 'Połącz konto, aby zobaczyć historię'}
        </p>
      </header>

      {!isLoggedIn ? (
        <Card className="border-2 border-dashed border-slate-200 bg-slate-50 shadow-none">
          <CardContent className="flex flex-col items-center justify-center gap-4 py-10 text-center">
            <div className="rounded-full bg-white p-3 text-emerald-500 shadow-sm">
              <LogIn size={24} />
            </div>
            <div>
              <h3 className="font-bold text-slate-700">Wymagane logowanie</h3>
              <p className="mt-1 max-w-[200px] text-xs text-slate-500">
                Aby pobrać Twoją historię słuchania, musisz połączyć konto Spotify.
              </p>
            </div>
            <Button
              onClick={handleConnect}
              className="gap-2 bg-emerald-500 text-white hover:bg-emerald-600"
            >
              Połącz ze Spotify <ExternalLink size={14} />
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {tracks.map((item: any, index: number) => {
            const isSelected = activeTrackId === item.track.id;

            return (
              <div key={`${item.track.id}-${index}`} className="flex flex-col gap-2">
                <Card
                  className={cn(
                    'overflow-hidden border-none bg-white shadow-sm transition-all',
                    isSelected && 'shadow-md ring-2 ring-emerald-500'
                  )}
                >
                  <CardContent className="flex items-center justify-between p-4">
                    <div className="flex items-center gap-3 overflow-hidden">
                      <div className="relative flex h-10 w-10 flex-shrink-0 items-center justify-center overflow-hidden rounded bg-slate-100">
                        {item.track.album.images?.[0] ? (
                          <img
                            src={item.track.album.images[2]?.url || item.track.album.images[0].url}
                            alt={item.track.name}
                            className="h-full w-full object-cover"
                          />
                        ) : (
                          <Music size={20} className="text-slate-400" />
                        )}
                      </div>
                      <div className="truncate">
                        <p className="truncate text-sm font-bold text-slate-800">
                          {item.track.name}
                        </p>
                        <p className="truncate text-[10px] text-slate-400">
                          {item.track.artists[0].name}
                        </p>
                      </div>
                    </div>
                    <Button
                      onClick={() => handleAnalyze(item.track)}
                      disabled={analyzingId === item.track.id}
                      size="sm"
                      variant={isSelected ? 'default' : 'secondary'}
                      className={cn(
                        'h-8 transition-colors',
                        isSelected
                          ? 'bg-emerald-500 text-white hover:bg-emerald-600'
                          : 'bg-slate-100 text-slate-600 hover:bg-emerald-100 hover:text-emerald-600'
                      )}
                    >
                      {analyzingId === item.track.id ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : isSelected ? (
                        'Ukryj'
                      ) : (
                        <Play size={14} />
                      )}
                    </Button>
                  </CardContent>
                </Card>

                {isSelected && results && (
                  <div className="animate-in slide-in-from-top-2 fade-in duration-300">
                    <Card className="gap-0 overflow-hidden border-none bg-white p-0 shadow-xl">
                      <div className="border-b border-emerald-100 bg-emerald-50 p-4">
                        <h3 className="flex items-center gap-2 text-sm font-bold text-emerald-800">
                          <Music size={16} /> Wynik analizy
                        </h3>
                      </div>
                      <CardContent className="p-6">
                        <div className="mb-6 flex w-full flex-col items-center">
                          <span className="mb-2 inline-block rounded-full bg-emerald-100 px-3 py-1 text-xs font-bold tracking-wider text-emerald-700 uppercase">
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
          })}
        </div>
      )}
    </div>
  );
}
