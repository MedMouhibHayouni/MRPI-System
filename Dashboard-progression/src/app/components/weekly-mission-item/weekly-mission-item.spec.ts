import { ComponentFixture, TestBed } from '@angular/core/testing';
import { WeeklyMissionItem } from './weekly-mission-item';

describe('WeeklyMissionItem', () => {
  let component: WeeklyMissionItem;
  let fixture: ComponentFixture<WeeklyMissionItem>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [WeeklyMissionItem],
    }).compileComponents();

    fixture = TestBed.createComponent(WeeklyMissionItem);
    fixture.componentRef.setInput('mission', {
      id: 'M-001',
      title: 'Compléter 3 exercices sur les dérivées',
      status: 'in_progress',
      deadline: '2026-07-15',
    });
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should not be marked as done when in progress', () => {
    expect(component.isDone()).toBe(false);
    expect(component.isInProgress()).toBe(true);
  });
});
