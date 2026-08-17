import { ApplicationConfig, importProvidersFrom, provideBrowserGlobalErrorListeners, provideZoneChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import {
  LucideAngularModule,
  AlertTriangle,
  ArrowRight,
  Award,
  BookOpen,
  CalendarClock,
  Check,
  CircleDashed,
  Clock,
  Dumbbell,
  Flame,
  Loader2,
  Medal,
  PlayCircle,
  Rocket,
  Sparkles,
  Star,
  Target,
  Trophy,
} from 'lucide-angular';

import { routes } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideRouter(routes),
    provideHttpClient(),
    importProvidersFrom(
      LucideAngularModule.pick({
        AlertTriangle,
        ArrowRight,
        Award,
        BookOpen,
        CalendarClock,
        Check,
        CircleDashed,
        Clock,
        Dumbbell,
        Flame,
        Loader2,
        Medal,
        PlayCircle,
        Rocket,
        Sparkles,
        Star,
        Target,
        Trophy,
      }),
    ),
  ],
};
