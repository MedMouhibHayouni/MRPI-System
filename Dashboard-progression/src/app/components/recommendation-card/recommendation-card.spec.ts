import { ComponentFixture, TestBed } from '@angular/core/testing';
import { RecommendationCard } from './recommendation-card';

describe('RecommendationCard', () => {
  let component: RecommendationCard;
  let fixture: ComponentFixture<RecommendationCard>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RecommendationCard],
    }).compileComponents();

    fixture = TestBed.createComponent(RecommendationCard);
    fixture.componentRef.setInput('rec', {
      id: 'R-01',
      title: 'Dérivées : règles de calcul',
      subject: 'Mathématiques',
      type: 'exercise',
      score: 0.94,
      difficulty: 'medium',
      estimated_minutes: 20,
    });
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should compute the score percentage', () => {
    expect(component.scorePct()).toBe(94);
  });
});
