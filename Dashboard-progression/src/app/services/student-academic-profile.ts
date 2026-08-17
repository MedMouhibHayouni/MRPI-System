import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { BehaviorSubject, Observable, of } from 'rxjs';
import { catchError, finalize, map, shareReplay, tap } from 'rxjs/operators';
import { StudentAcademicProfile } from '../models/student-profile.model';
import { ApiErrorHandler } from './api-error-handler';

const CACHE_TTL_MS = 5 * 60 * 1000;

/**
 * Loads the real learner profiles bundled from students.csv (public/students.json).
 * This is dataset data, not gamification mock data — it drives the actual
 * POST /recommendations request body (student_id, subject, weak_concept,
 * academic_level, learning_style, past_interactions) against the FastAPI backend.
 * Cached for 5 minutes (CDC 4.4.3), then re-fetched on the next call.
 */
@Injectable({ providedIn: 'root' })
export class StudentAcademicProfileService {
  private readonly http = inject(HttpClient);
  private readonly errorHandler = inject(ApiErrorHandler);
  private readonly assetUrl = '/students.json';

  private readonly _profiles$ = new BehaviorSubject<StudentAcademicProfile[]>([]);
  private readonly _loading$ = new BehaviorSubject<boolean>(false);
  private readonly _errorMessage$ = new BehaviorSubject<string | null>(null);

  readonly profiles$ = this._profiles$.asObservable();
  readonly loading$ = this._loading$.asObservable();
  readonly errorMessage$ = this._errorMessage$.asObservable();

  private request$: Observable<StudentAcademicProfile[]> | null = null;
  private lastFetchedAt = 0;

  /** Fetches the full profile list, cached for 5 minutes. */
  loadProfiles(forceRefresh = false): Observable<StudentAcademicProfile[]> {
    const isCacheFresh = Date.now() - this.lastFetchedAt < CACHE_TTL_MS;
    if (!forceRefresh && isCacheFresh && this.request$) {
      return this.request$;
    }

    this._loading$.next(true);
    this._errorMessage$.next(null);

    this.request$ = this.http.get<StudentAcademicProfile[]>(this.assetUrl).pipe(
      tap((profiles) => {
        this._profiles$.next(profiles);
        this.lastFetchedAt = Date.now();
      }),
      catchError((err: HttpErrorResponse) => {
        this._errorMessage$.next(this.errorHandler.toUserMessage(err));
        return of([] as StudentAcademicProfile[]);
      }),
      finalize(() => this._loading$.next(false)),
      shareReplay(1),
    );

    return this.request$;
  }

  /** Looks up a single profile by student_id from the already-loaded list. */
  getProfile(studentId: string): Observable<StudentAcademicProfile | undefined> {
    return this.profiles$.pipe(map((profiles) => profiles.find((p) => p.student_id === studentId)));
  }
}