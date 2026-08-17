import { ComponentFixture, TestBed } from '@angular/core/testing';
import { XpProgressBar } from './xp-progress-bar';

describe('XpProgressBar', () => {
  let component: XpProgressBar;
  let fixture: ComponentFixture<XpProgressBar>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [XpProgressBar],
    }).compileComponents();

    fixture = TestBed.createComponent(XpProgressBar);
    fixture.componentRef.setInput('item', {
      subject: 'Mathématiques',
      xp: 340,
      xp_next: 500,
      percentage: 0.68,
    });
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should compute the rounded percentage', () => {
    expect(component.pct()).toBe(68);
  });
});
