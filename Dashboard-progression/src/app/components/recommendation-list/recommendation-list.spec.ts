import { ComponentFixture, TestBed } from '@angular/core/testing';
import { RecommendationList } from './recommendation-list';

describe('RecommendationList', () => {
  let component: RecommendationList;
  let fixture: ComponentFixture<RecommendationList>;

  const recs = [
    { id: 'R-01', title: 'A', subject: 'Maths', type: 'exercise' as const, score: 0.5, difficulty: 'easy' as const, estimated_minutes: 30 },
    { id: 'R-02', title: 'B', subject: 'Maths', type: 'video' as const, score: 0.9, difficulty: 'easy' as const, estimated_minutes: 10 },
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RecommendationList],
    }).compileComponents();

    fixture = TestBed.createComponent(RecommendationList);
    fixture.componentRef.setInput('recommendations', recs);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should sort by score descending by default', () => {
    expect(component.items().map((r) => r.resource_id)).toEqual(['R-02', 'R-01']);
  });

  it('should filter by type', () => {
    component.onFilterChange('video');
    expect(component.items().map((r) => r.resource_id)).toEqual(['R-02']);
  });

  it('should sort by duration when selected', () => {
    component.onSortChange('duration');
    expect(component.items().map((r) => r.resource_id)).toEqual(['R-02', 'R-01']);
  });
});
