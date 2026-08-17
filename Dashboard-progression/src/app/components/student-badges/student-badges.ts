import { Component, input } from '@angular/core';
import { LucideAngularModule } from 'lucide-angular';
import { Badge } from '../../models/student.model';

/**
 * Simple list of earned badges, each rendered as a pill with an icon.
 * Renders nothing when the student has no badges yet.
 */
@Component({
  selector: 'app-student-badges',
  standalone: true,
  imports: [LucideAngularModule],
  templateUrl: './student-badges.html',
  styleUrl: './student-badges.css',
})
export class StudentBadges {
  badges = input.required<Badge[]>();
}
