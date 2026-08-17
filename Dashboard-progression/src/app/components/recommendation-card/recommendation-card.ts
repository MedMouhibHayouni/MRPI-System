import { Component, computed, input } from '@angular/core';
import { NgClass } from '@angular/common';
import { LucideAngularModule } from 'lucide-angular';
import { Difficulty, Recommendation, RecommendationType } from '../../models/recommendation.model';

interface TypeMeta {
  icon: string;
  label: string;
  cls: string;
}

interface DiffMeta {
  label: string;
  cls: string;
}

const TYPE_META: Record<RecommendationType, TypeMeta> = {
  exercise: { icon: 'dumbbell', label: 'Exercice', cls: 'bg-primary/10 text-primary' },
  micro_lesson: { icon: 'book-open', label: 'Micro-leçon', cls: 'bg-success/15 text-success' },
  video: { icon: 'play-circle', label: 'Vidéo', cls: 'bg-accent text-accent-foreground' },
};

const DIFF_META: Record<Difficulty, DiffMeta> = {
  beginner: { label: 'Facile', cls: 'bg-success/15 text-success border-success/30' },
  intermediate: { label: 'Moyen', cls: 'bg-warning/20 text-warning-foreground border-warning/40' },
  advanced: { label: 'Difficile', cls: 'bg-destructive/15 text-destructive border-destructive/30' },
};

@Component({
  selector: 'app-recommendation-card',
  standalone: true,
  imports: [NgClass, LucideAngularModule],
  templateUrl: './recommendation-card.html',
  styleUrl: './recommendation-card.css',
})
export class RecommendationCard {
  rec = input.required<Recommendation>();

  typeMeta = computed(() => TYPE_META[this.rec().type]);
  diffMeta = computed(() => DIFF_META[this.rec().difficulty]);
  scorePct = computed(() => Math.round(this.rec().relevance_score * 100));
}