import { WhoopAdapter } from './whoop';
import { OuraAdapter } from './oura';
import { GarminAdapter } from './garmin';
import type { WearableAdapter } from './types';

export const wearableAdapters: Record<string, WearableAdapter> = {
  whoop: new WhoopAdapter(),
  oura: new OuraAdapter(),
  garmin: new GarminAdapter(),
};

export * from './types';
