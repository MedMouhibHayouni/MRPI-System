import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { BehaviorSubject, Observable, of } from 'rxjs';
import { catchError, finalize, tap } from 'rxjs/operators';
import { StudentProfile } from '../models/student.model';
import { ApiErrorHandler } from './api-error-handler';
import { environment } from '../../environments/environment';

const CACHE_TTL_MS = 5 * 60 * 1000;   /* 5 minutes */

/**
 * Global-state service for the student identity block (name/level/xp/badges).
 * Mock-sourced (db.json, single object) — no backend endpoint exists for
 * this, so it stays constant regardless of the selected student_id.
 */
@Injectable({ providedIn: 'root' })
export class StudentService {
  private readonly http = inject(HttpClient);
  private readonly errorHandler = inject(ApiErrorHandler);
  private readonly apiUrl = `${environment.mockApiUrl}/student`;

  private readonly _student$ = new BehaviorSubject<StudentProfile | null>(null);
  private readonly _loading$ = new BehaviorSubject<boolean>(false);
  private readonly _errorMessage$ = new BehaviorSubject<string | null>(null);

  readonly student$ = this._student$.asObservable();
  readonly loading$ = this._loading$.asObservable();
  readonly errorMessage$ = this._errorMessage$.asObservable();

  private lastFetchedAt = 0;

  getStudent(forceRefresh = false): Observable<StudentProfile | null> {
    const isCacheFresh = Date.now() - this.lastFetchedAt < CACHE_TTL_MS;
    if (!forceRefresh && isCacheFresh && this._student$.value) {
      return this.student$;
    }

    this._loading$.next(true);
    this._errorMessage$.next(null);

    this.http
      .get<StudentProfile>(this.apiUrl)
      .pipe(
        tap((student) => {
          this._student$.next(student);
          this.lastFetchedAt = Date.now();
        }),
        catchError((err: HttpErrorResponse) => {
          this._errorMessage$.next(this.errorHandler.toUserMessage(err));
          return of(null);
        }),
        finalize(() => this._loading$.next(false)),
      )
      .subscribe();

    return this.student$;
  }
}