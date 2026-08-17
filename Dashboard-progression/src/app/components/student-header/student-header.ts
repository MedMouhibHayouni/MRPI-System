import { Component, computed, input } from '@angular/core';
import { LucideAngularModule } from 'lucide-angular';
import { StudentProfile } from '../../models/student.model';
import { StudentBadges } from '../student-badges/student-badges';

@Component({
  selector: 'app-student-header',
  standalone: true,
  imports: [LucideAngularModule, StudentBadges],
  templateUrl: './student-header.html',
  styleUrl: './student-header.css',
})
export class StudentHeader {
  student = input.required<StudentProfile>();

  initials = computed(() =>
    this.student()
      .name.split(' ')
      .map((n) => n[0])
      .slice(0, 2)
      .join(''),
  );

  formattedXp = computed(() => this.student().total_xp.toLocaleString('fr-FR'));
}
