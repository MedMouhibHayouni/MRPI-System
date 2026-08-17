import { ComponentFixture, TestBed } from '@angular/core/testing';
import { StudentHeader } from './student-header';

describe('StudentHeader', () => {
  let component: StudentHeader;
  let fixture: ComponentFixture<StudentHeader>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [StudentHeader],
    }).compileComponents();

    fixture = TestBed.createComponent(StudentHeader);
    fixture.componentRef.setInput('student', {
      id: 'STU-2026-0042',
      name: 'Amira Ben Salah',
      level: 12,
      avatar_url: '',
      total_xp: 3420,
    });
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should compute initials from the student name', () => {
    expect(component.initials()).toBe('AB');
  });
});
