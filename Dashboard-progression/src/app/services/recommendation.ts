import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { BehaviorSubject, Observable, of } from 'rxjs';
import { catchError, finalize, tap } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import { RecommendationRequest, RecommendationsResponse } from '../models/recommendation.model';
import { ApiErrorHandler } from './api-error-handler';

const CACHE_TTL_MS = 5 * 60 * 1000;

/**
 * Global-state service for the recommendations list.
 * Real backend: POST http://localhost:8000/recommendations — the actual
 * CBF+CF hybrid engine, fed the full learner profile (subject, weak_concept,
 * academic_level, learning_style, past_interactions). This is intentionally
 * NOT mocked: the recommendations, their scores, and the LLM explanation
 * all depend on that profile, not on the student_id alone.
 * Cache is keyed on the serialized request body, not just time, since two
 * different profiles must never share a cached result.
 */
@Injectable({ providedIn: 'root' })
export class RecommendationService {
  private readonly http = inject(HttpClient);
  private readonly errorHandler = inject(ApiErrorHandler);
  private readonly apiUrl = `${environment.apiUrl}/recommendations`;

  private readonly _recommendations$ = new BehaviorSubject<RecommendationsResponse | null>(null);
  private readonly _loading$ = new BehaviorSubject<boolean>(false);
  private readonly _errorMessage$ = new BehaviorSubject<string | null>(null);

  readonly recommendations$ = this._recommendations$.asObservable();
  readonly loading$ = this._loading$.asObservable();
  readonly errorMessage$ = this._errorMessage$.asObservable();

  private lastFetchedAt = 0;
  private lastRequestKey: string | null = null;

  getRecommendations(
    request: RecommendationRequest,
    forceRefresh = false,
  ): Observable<RecommendationsResponse | null> {
    const requestKey = JSON.stringify(request);
    const isCacheFresh = Date.now() - this.lastFetchedAt < CACHE_TTL_MS;
    if (!forceRefresh && isCacheFresh && requestKey === this.lastRequestKey && this._recommendations$.value) {
      return this.recommendations$;
    }

    this._loading$.next(true);
    this._errorMessage$.next(null);

    this.http
      .post<RecommendationsResponse>(this.apiUrl, request)
      .pipe(
        tap((recs) => {
          this._recommendations$.next(recs);
          this.lastFetchedAt = Date.now();
          this.lastRequestKey = requestKey;
        }),
        catchError((err: HttpErrorResponse) => {
          this._errorMessage$.next(this.errorHandler.toUserMessage(err));
          return of(null);
        }),
        finalize(() => this._loading$.next(false)),
      )
      .subscribe();

    return this.recommendations$;
  }
}
