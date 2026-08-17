import { ComponentFixture, TestBed } from '@angular/core/testing';
import { WeeklyMissionList } from './weekly-mission-list';

describe('WeeklyMissionList', () => {
  let component: WeeklyMissionList;
  let fixture: ComponentFixture<WeeklyMissionList>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [WeeklyMissionList],
    }).compileComponents();

    fixture = TestBed.createComponent(WeeklyMissionList);
    fixture.componentRef.setInput('missions', [
      { id: 'M-001', title: 'A', status: 'completed', deadline: '2026-07-15' },
      { id: 'M-002', title: 'B', status: 'not_started', deadline: '2026-07-17' },
    ]);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should compute completion percentage', () => {
    expect(component.done()).toBe(1);
    expect(component.pct()).toBe(50);
  });
});
