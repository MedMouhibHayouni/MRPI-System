import { ComponentFixture, TestBed } from '@angular/core/testing';
import { StudentBadges } from './student-badges';

describe('StudentBadges', () => {
  let component: StudentBadges;
  let fixture: ComponentFixture<StudentBadges>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [StudentBadges],
    }).compileComponents();

    fixture = TestBed.createComponent(StudentBadges);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    fixture.componentRef.setInput('badges', []);
    fixture.detectChanges();
    expect(component).toBeTruthy();
  });

  it('should render one pill per badge', () => {
    fixture.componentRef.setInput('badges', [
      { id: 'streak-7', label: 'Série de 7 jours', icon: 'flame' },
      { id: 'top-math', label: 'Champion en Mathématiques', icon: 'award' },
    ]);
    fixture.detectChanges();

    const pills = fixture.nativeElement.querySelectorAll('[role="listitem"]');
    expect(pills.length).toBe(2);
    expect(pills[0].textContent).toContain('Série de 7 jours');
  });

  it('should render nothing when there are no badges', () => {
    fixture.componentRef.setInput('badges', []);
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('[role="list"]')).toBeNull();
  });
});
