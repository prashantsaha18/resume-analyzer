import axios from 'axios';
import { getIdToken } from './firebase';

const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
const api = axios.create({ baseURL: BASE_URL });

api.interceptors.request.use(async (config) => {
  const token = await getIdToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg = err.response?.data?.error || err.response?.data?.detail || err.message;
    return Promise.reject(new Error(msg));
  }
);

// Users
export const syncUser       = ()     => api.post('/users/sync/');
export const getProfile     = ()     => api.get('/users/profile/');
export const updateProfile  = (d)    => api.patch('/users/profile/', d);
export const getDashboardStats = ()  => api.get('/users/dashboard/');

// Resumes
export const getResumes      = ()    => api.get('/resumes/');
export const getResume       = (id)  => api.get(`/resumes/${id}/`);
export const getPrimaryResume= ()    => api.get('/resumes/primary/');
export const createResume    = (d)   => api.post('/resumes/', d);
export const updateResume    = (id,d)=> api.patch(`/resumes/${id}/`, d);
export const deleteResume    = (id)  => api.delete(`/resumes/${id}/`);
export const duplicateResume = (id)  => api.post(`/resumes/${id}/duplicate/`);
export const setPrimaryResume= (id)  => api.post(`/resumes/${id}/set-primary/`);
export const uploadResumePDF = (fd)  => api.post('/resumes/upload-pdf/', fd, {headers:{'Content-Type':'multipart/form-data'}});

// Analysis
export const analyzeResume    = (d)  => api.post('/analysis/analyze/', d);
export const atsKeywordMatch  = (d)  => api.post('/analysis/ats-match/', d);
export const skillGapAnalysis = (d)  => api.post('/analysis/skill-gap/', d);
export const getAnalysisHistory=(id) => api.get('/analysis/history/', {params:{resume_id:id}});

// AI Features
export const improveResume       = (d) => api.post('/ai/improve-resume/', d);
export const enhanceBullet       = (d) => api.post('/ai/enhance-bullet/', d);
export const detectWeaknesses    = (d) => api.post('/ai/detect-weaknesses/', d);
export const checkGrammar        = (d) => api.post('/ai/check-grammar/', d);
export const generateCoverLetter = (d) => api.post('/ai/cover-letter/', d);
export const recommendJobRoles   = (d) => api.post('/ai/job-roles/', d);
export const jobApplyAssistant   = (d) => api.post('/ai/job-apply/', d);
export const detectFakeResume    = (d) => api.post('/ai/fake-detect/', d);
export const generatePortfolio   = (d) => api.post('/ai/portfolio/', d);

// Chat
export const getConversations    = ()       => api.get('/chat/conversations/');
export const createConversation  = (title)  => api.post('/chat/conversations/', {title});
export const getConversation     = (id)     => api.get(`/chat/conversations/${id}/`);
export const deleteConversation  = (id)     => api.delete(`/chat/conversations/${id}/`);
export const sendMessage         = (id, msg)=> api.post(`/chat/conversations/${id}/message/`, {message:msg});
export const quickChat           = (msg,ctx)=> api.post('/chat/quick/', {message:msg, context:ctx});

export default api;