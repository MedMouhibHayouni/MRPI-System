import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { ProgressService } from './progress';
import { ProgressResponse } from '../models/progress.model';

describe('ProgressService', () => {
  let service: ProgressService;
  let httpMock: HttpTestingController;

  const mock: ProgressResponse = {
    student_id: 'STU-2026-0042',
    subject_mastery: [],
    weekly_missions: [],
  };

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(ProgressService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('fetches progress data and pushes it into the shared state', () => {
    let emitted: ProgressResponse | null = null;
    service.progress$.subscribe((p) => (emitted = p));

    service.getProgress().subscribe();

    const req = httpMock.expectOne('http://localhost:3000/progress');
    expect(req.request.method).toBe('GET');
    req.flush(mock);

    expect(emitted!).toEqual(mock);
  });

  it('serves subsequent calls from the in-memory cache instead of refetching', () => {
    service.getProgress().subscribe();
    httpMock.expectOne('http://localhost:3000/progress').flush(mock);

    service.getProgress().subscribe();
    httpMock.expectNone('http://localhost:3000/progress');
  });
});
