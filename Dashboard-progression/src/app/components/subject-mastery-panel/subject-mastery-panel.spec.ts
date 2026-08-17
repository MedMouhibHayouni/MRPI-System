import { ComponentFixture, TestBed } from '@angular/core/testing';
import { SubjectMasteryPanel } from './subject-mastery-panel';

describe('SubjectMasteryPanel', () => {
  let component: SubjectMasteryPanel;
  let fixture: ComponentFixture<SubjectMasteryPanel>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SubjectMasteryPanel],
    }).compileComponents();

    fixture = TestBed.createComponent(SubjectMasteryPanel);
    fixture.componentRef.setInput('items', [
      { subject: 'Mathématiques', xp: 340, xp_next: 500, percentage: 0.68 },
      { subject: 'Physique', xp: 210, xp_next: 300, percentage: 0.7 },
    ]);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should compute the average mastery percentage', () => {
    expect(component.avg()).toBe(69);
  });
});
