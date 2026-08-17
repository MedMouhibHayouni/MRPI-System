import { Component, computed, input, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { Recommendation, RecommendationType } from '../../models/recommendation.model';
import { RecommendationCard } from '../recommendation-card/recommendation-card';

type SortBy = 'score' | 'duration';
type Filter = 'all' | RecommendationType;

@Component({
  selector: 'app-recommendation-list',
  standalone: true,
  imports: [FormsModule, LucideAngularModule, RecommendationCard],
  templateUrl: './recommendation-list.html',
  styleUrl: './recommendation-list.css',
})
export class RecommendationList {
  recommendations = input.required<Recommendation[]>();
  explanation = input<string>();

  filter = signal<Filter>('all');
  sort = signal<SortBy>('score');

  items = computed(() => {
    const filter = this.filter();
    const sort = this.sort();
    const filtered =
      filter === 'all' ? this.recommendations() : this.recommendations().filter((r) => r.type === filter);
    return [...filtered].sort((a, b) =>
      sort === 'score' ? b.relevance_score - a.relevance_score : a.estimated_time_min - b.estimated_time_min,
    );
  });

  onFilterChange(value: string): void {
    this.filter.set(value as Filter);
  }

  onSortChange(value: string): void {
    this.sort.set(value as SortBy);
  }
}
