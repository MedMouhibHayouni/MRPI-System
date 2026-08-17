import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { RecommendationService } from './recommendation';
import { RecommendationRequest, RecommendationsResponse } from '../models/recommendation.model';

describe('RecommendationService', () => {
  let service: RecommendationService;
  let httpMock: HttpTestingController;

  const request: RecommendationRequest = {
    student_id: 'STU-2026-0001',
    subject: 'Mathematiques',
    weak_concept: 'fractions',
    academic_level: 'beginner',
    learning_style: 'visual',
    past_interactions: ['RES-001'],
  };

  const mock: RecommendationsResponse = {
    request_id: 'req-1',
    student_id: 'STU-2026-0001',
    recommendations: [],
    llm_explanation: 'test',
    metadata: { latency_ms: 42 },
  };

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(RecommendationService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('posts the full profile to the real backend and pushes the result into shared state', () => {
    let emitted: RecommendationsResponse | null = null;
    service.recommendations$.subscribe((r) => (emitted = r));

    service.getRecommendations(request).subscribe();

    const req = httpMock.expectOne('http://localhost:8000/recommendations');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual(request);
    req.flush(mock);

    expect(emitted!).toEqual(mock);
  });

  it('serves subsequent calls for the same profile from the in-memory cache instead of refetching', () => {
    service.getRecommendations(request).subscribe();
    httpMock.expectOne('http://localhost:8000/recommendations').flush(mock);

    service.getRecommendations(request).subscribe();
    httpMock.expectNone('http://localhost:8000/recommendations');
  });
});
