import { initializeApp } from 'firebase/app';
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword,
         signInWithPopup, GoogleAuthProvider, signOut, sendPasswordResetEmail,
         onAuthStateChanged, updateProfile } from 'firebase/auth';

const firebaseConfig = {
  apiKey:            process.env.REACT_APP_FIREBASE_API_KEY,
  authDomain:        process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
  projectId:         process.env.REACT_APP_FIREBASE_PROJECT_ID,
  storageBucket:     process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
  appId:             process.env.REACT_APP_FIREBASE_APP_ID,
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
const googleProvider = new GoogleAuthProvider();

export const loginWithEmail    = (e, p)    => signInWithEmailAndPassword(auth, e, p);
export const registerWithEmail = async (e, p, name) => {
  const cred = await createUserWithEmailAndPassword(auth, e, p);
  if (name) await updateProfile(cred.user, { displayName: name });
  return cred;
};
export const loginWithGoogle = () => signInWithPopup(auth, googleProvider);
export const logout          = () => signOut(auth);
export const resetPassword   = (e) => sendPasswordResetEmail(auth, e);
export const onAuthChange    = (cb) => onAuthStateChanged(auth, cb);
export const getIdToken      = async () => auth.currentUser?.getIdToken() || null;
export default app;