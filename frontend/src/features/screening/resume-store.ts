// Stores uploaded resume PDFs in the browser (IndexedDB) so previews survive reloads.
const DB = "shortlist-resumes";
const STORE = "files";

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx<T>(
  mode: IDBTransactionMode,
  fn: (s: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE, mode);
    const request = fn(transaction.objectStore(STORE));
    request.onerror = () => reject(request.error);
    transaction.oncomplete = () => {
      db.close();
      resolve(request.result);
    };
    transaction.onabort = () => {
      db.close();
      reject(transaction.error ?? request.error ?? new Error("IndexedDB transaction aborted."));
    };
  });
}

export const putResume = (id: string, file: Blob) =>
  tx("readwrite", (s) => s.put(file, id)).catch(() => undefined);
export const getResume = (id: string) =>
  tx<Blob | undefined>("readonly", (s) => s.get(id)).catch(() => undefined);
export const deleteResumes = async (ids: string[]) => {
  for (const id of ids) await tx("readwrite", (s) => s.delete(id));
};
export const clearResumes = () => tx("readwrite", (s) => s.clear());
