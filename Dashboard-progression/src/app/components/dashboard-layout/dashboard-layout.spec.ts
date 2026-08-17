import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { DashboardLayout } from './dashboard-layout';

describe('DashboardLayout', () => {
  let component: DashboardLayout;
  let fixture: ComponentFixture<DashboardLayout>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DashboardLayout],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(DashboardLayout);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    fixture.detectChanges();
  });

  afterEach(() => {
    httpMock.verify();
  });

  function flushMockData(): void {
    httpMock.expectOne('http://localhost:3000/student').flush({
      id: 'STU-2026-0042',
      name: 'Amira Ben Salah',
      level: 12,
      avatar_url: '',
      total_xp: 3420,
    });
    httpMock.expectOne('http://localhost:3000/recommendations').flush({
      request_id: 'req-1',
      recommendations: [],
      llm_explanation: 'test',
    });
    httpMock.expectOne('http://localhost:3000/progress').flush({
      student_id: 'STU-2026-0042',
      subject_mastery: [],
      weekly_missions: [],
    });
    fixture.detectChanges();
  }

  it('should create', () => {
    flushMockData();
    expect(component).toBeTruthy();
  });

  it('should start in a loading state before the requests resolve', () => {
    expect(component.isLoading()).toBe(true);
    flushMockData();
  });

  it('should stop loading once all three resources resolve', () => {
    flushMockData();
    expect(component.isLoading()).toBe(false);
    expect(component.hasError()).toBe(false);
  });

  it('should default to the recommendations tab', () => {
    flushMockData();
    expect(component.activeTab()).toBe('recs');
  });

  it('should switch tabs', () => {
    flushMockData();
    component.setTab('progress');
    expect(component.activeTab()).toBe('progress');
  });
});
