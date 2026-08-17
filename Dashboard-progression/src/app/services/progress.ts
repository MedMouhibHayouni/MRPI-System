import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { BehaviorSubject, Observable, of } from 'rxjs';
import { catchError, finalize, tap } from 'rxjs/operators';
import { ProgressResponse } from '../models/progress.model';
import { ApiErrorHandler } from './api-error-handler';
import { environment } from '../../environments/environment';

const CACHE_TTL_MS = 5 * 60 * 1000;

/**
 * Global-state service for progression data (mastery + missions).
 * Mock-sourced (db.json, single object) — no backend endpoint exists for
 * this, so it stays constant regardless of the selected student_id.
 */
@Injectable({ providedIn: 'root' })
export class ProgressService {
  private readonly http = inject(HttpClient);
  private readonly errorHandler = inject(ApiErrorHandler);
  private readonly apiUrl = `${environment.mockApiUrl}/progress`;

  private readonly _progress$ = new BehaviorSubject<ProgressResponse | null>(null);
  private readonly _loading$ = new BehaviorSubject<boolean>(false);
  private readonly _errorMessage$ = new BehaviorSubject<string | null>(null);

  readonly progress$ = this._progress$.asObservable();
  readonly loading$ = this._loading$.asObservable();
  readonly errorMessage$ = this._errorMessage$.asObservable();

  private lastFetchedAt = 0;

  getProgress(forceRefresh = false): Observable<ProgressResponse | null> {
    const isCacheFresh = Date.now() - this.lastFetchedAt < CACHE_TTL_MS;
    if (!forceRefresh && isCacheFresh && this._progress$.value) {
      return this.progress$;
    }

    this._loading$.next(true);
    this._errorMessage$.next(null);

    this.http
      .get<ProgressResponse>(this.apiUrl)
      .pipe(
        tap((progress) => {
          this._progress$.next(progress);
          this.lastFetchedAt = Date.now();
        }),
        catchError((err: HttpErrorResponse) => {
          this._errorMessage$.next(this.errorHandler.toUserMessage(err));
          return of(null);
        }),
        finalize(() => this._loading$.next(false)),
      )
      .subscribe();

    return this.progress$;
  }
}