import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { StudentService } from './student';
import { StudentProfile } from '../models/student.model';

describe('StudentService', () => {
  let service: StudentService;
  let httpMock: HttpTestingController;

  const mock: StudentProfile = {
    id: 'STU-2026-0042',
    name: 'Amira Ben Salah',
    level: 12,
    avatar_url: '',
    total_xp: 3420,
  };

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(StudentService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('fetches the student profile and pushes it into the shared state', () => {
    let emitted: StudentProfile | null = null;
    service.student$.subscribe((s) => (emitted = s));

    service.getStudent().subscribe();

    const req = httpMock.expectOne('http://localhost:3000/student');
    expect(req.request.method).toBe('GET');
    req.flush(mock);

    expect(emitted!).toEqual(mock);
  });

  it('serves subsequent calls from the in-memory cache instead of refetching', () => {
    service.getStudent().subscribe();
    httpMock.expectOne('http://localhost:3000/student').flush(mock);

    service.getStudent().subscribe();
    httpMock.expectNone('http://localhost:3000/student');
  });

  it('sets a user-friendly error message on failure', () => {
    let error: string | null = null;
    service.errorMessage$.subscribe((e) => (error = e));

    service.getStudent().subscribe();
    const req = httpMock.expectOne('http://localhost:3000/student');
    req.flush('fail', { status: 500, statusText: 'Server Error' });

    expect(error).toContain('serveur');
  });
});
