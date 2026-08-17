import { Component, computed, effect, inject, signal } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { LucideAngularModule } from 'lucide-angular';

import { StudentService } from '../../services/student';
import { RecommendationService } from '../../services/recommendation';
import { ProgressService } from '../../services/progress';
import { StudentAcademicProfileService } from '../../services/student-academic-profile';

import { StudentHeader } from '../student-header/student-header';
import { SubjectMasteryPanel } from '../subject-mastery-panel/subject-mastery-panel';
import { RecommendationList } from '../recommendation-list/recommendation-list';
import { WeeklyMissionList } from '../weekly-mission-list/weekly-mission-list';

type Tab = 'recs' | 'progress' | 'missions';

@Component({
  selector: 'app-dashboard-layout',
  standalone: true,
  imports: [LucideAngularModule, StudentHeader, SubjectMasteryPanel, RecommendationList, WeeklyMissionList],
  templateUrl: './dashboard-layout.html',
  styleUrl: './dashboard-layout.css',
})
export class DashboardLayout {
  private readonly studentService = inject(StudentService);
  private readonly recommendationService = inject(RecommendationService);
  private readonly progressService = inject(ProgressService);
  private readonly profileService = inject(StudentAcademicProfileService);

  // Data sourcing split:
  // - student (name/level/avatar/xp/badges): mock db.json — the FastAPI backend has no
  //   endpoint for this, so it stays constant regardless of the selected student_id.
  // - progress (subject_mastery/weekly_missions): mock db.json — same reason, constant.
  // - recs: REAL FastAPI POST http://localhost:8000/recommendations, fed the full learner
  //   profile (subject/weak_concept/academic_level/learning_style/past_interactions) loaded
  //   from public/students.json (students.csv), keyed by the selected student_id. This is
  //   the only piece that changes when you switch students, and it changes because the
  //   CBF+CF engine is genuinely fed different input each time — not because the id alone
  //   fetches something.
  student = toSignal(this.studentService.student$, { initialValue: null });
  recs = toSignal(this.recommendationService.recommendations$, { initialValue: null });
  progress = toSignal(this.progressService.progress$, { initialValue: null });
  profiles = toSignal(this.profileService.profiles$, { initialValue: [] });

  private studentLoading = toSignal(this.studentService.loading$, { initialValue: false });
  private recsLoading = toSignal(this.recommendationService.loading$, { initialValue: false });
  private progressLoading = toSignal(this.progressService.loading$, { initialValue: false });
  private profilesLoading = toSignal(this.profileService.loading$, { initialValue: false });

  private studentError = toSignal(this.studentService.errorMessage$, { initialValue: null });
  private recsError = toSignal(this.recommendationService.errorMessage$, { initialValue: null });
  private progressError = toSignal(this.progressService.errorMessage$, { initialValue: null });
  private profilesError = toSignal(this.profileService.errorMessage$, { initialValue: null });

  isLoading = computed(
    () =>
      this.studentLoading() || this.recsLoading() || this.progressLoading() || this.profilesLoading(),
  );

  errorMessage = computed(
    () => this.studentError() ?? this.recsError() ?? this.progressError() ?? this.profilesError(),
  );

  /**
   * True once loading has settled and something is still wrong (explicit error, or a
   * resource resolved to null/empty). Drives the skeleton fallback in the template —
   * intentionally NOT surfaced as an error card in the UI; see the console.error effect
   * below. Known gap vs CDC 4.4.3 ("messages user-friendly"), kept as a deliberate
   * decision for this milestone rather than an oversight.
   */
  hasError = computed(
    () =>
      !this.isLoading() &&
      (!!this.errorMessage() || !this.student() || !this.recs() || !this.progress()),
  );

  /** Real student_id driving the live /recommendations call — independent from the mocked student header. */
  selectedStudentId = signal<string | null>(null);

  activeTab = signal<Tab>('recs');
  private readonly tabOrder: Tab[] = ['recs', 'progress', 'missions'];

  constructor() {
    // Errors surface here only — never in the UI (see hasError doc comment).
    effect(() => {
      const err = this.errorMessage();
      if (err) {
        console.error('[DashboardLayout] data fetch failed:', err);
      } else if (!this.isLoading() && this.hasError()) {
        console.error(
          '[DashboardLayout] load settled with missing data (student/recs/progress null) — check mock API / FastAPI backend.',
        );
      }
    });

    this.studentService.getStudent().subscribe();
    this.progressService.getProgress().subscribe();

    this.profileService.loadProfiles().subscribe((profiles) => {
      if (profiles.length && !this.selectedStudentId()) {
        this.selectedStudentId.set(profiles[0].student_id);
      }
    });

    // Re-fetches from the real API every time the selection changes (or profiles finish loading).
    effect(() => {
      const studentId = this.selectedStudentId();
      const profile = this.profiles().find((p) => p.student_id === studentId);
      if (!profile) {
        return;
      }
      this.recommendationService
        .getRecommendations({
          student_id: profile.student_id,
          subject: profile.subject,
          weak_concept: profile.weak_concept,
          academic_level: profile.academic_level,
          learning_style: profile.learning_style,
          past_interactions: profile.past_interactions,
        })
        .subscribe();
    });
  }

  onStudentIdChange(studentId: string): void {
    this.selectedStudentId.set(studentId);
  }

  setTab(tab: Tab): void {
    this.activeTab.set(tab);
  }

  /** Arrow-key navigation for the mobile tablist (WAI-ARIA tabs pattern, per 4.3.4). */
  onTabKeydown(event: KeyboardEvent): void {
    const currentIndex = this.tabOrder.indexOf(this.activeTab());
    let nextIndex: number | null = null;

    if (event.key === 'ArrowRight') {
      nextIndex = (currentIndex + 1) % this.tabOrder.length;
    } else if (event.key === 'ArrowLeft') {
      nextIndex = (currentIndex - 1 + this.tabOrder.length) % this.tabOrder.length;
    } else if (event.key === 'Home') {
      nextIndex = 0;
    } else if (event.key === 'End') {
      nextIndex = this.tabOrder.length - 1;
    }

    if (nextIndex === null) {
      return;
    }

    event.preventDefault();
    const nextTab = this.tabOrder[nextIndex];
    this.activeTab.set(nextTab);
    queueMicrotask(() => {
      const el = (event.currentTarget as HTMLElement).querySelector<HTMLElement>(
        `#tab-${nextTab}`,
      );
      el?.focus();
    });
  }
}