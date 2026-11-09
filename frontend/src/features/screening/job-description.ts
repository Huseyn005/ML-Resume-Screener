export const JOB_DESCRIPTION_MIN_LENGTH = 50;
export const JOB_DESCRIPTION_MAX_LENGTH = 10_000;

export function jobDescriptionLength(value: string): number {
  return Array.from(value.trim()).length;
}

export function isJobDescriptionValid(value: string): boolean {
  const length = jobDescriptionLength(value);
  return length >= JOB_DESCRIPTION_MIN_LENGTH && length <= JOB_DESCRIPTION_MAX_LENGTH;
}
