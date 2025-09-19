import React, { useState, useEffect } from 'react';
import './UploadDocument.css';


// Progress Bar Component
const ProgressBar = ({ isIndeterminate = true, progress = 0, label = "Processing..." }) => (
  <div className="progress-container">
    <div className="progress-label">{label}</div>
    <div className="progress-bar-wrapper">
      <div 
        className={`progress-bar ${isIndeterminate ? 'indeterminate' : ''}`}
        style={!isIndeterminate ? { width: `${progress}%` } : {}}
      />
    </div>
    {!isIndeterminate && <span className="progress-percentage">{progress}%</span>}
  </div>
);

// Error Modal Component
const ErrorModal = ({ isOpen, onClose, error, title = "Error" }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>{title}</h3>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          <div className="error-icon">⚠️</div>
          <div className="error-message">
            {typeof error === 'string' ? error : error?.message || 'An unexpected error occurred'}
          </div>
          {error?.details && (
            <details className="error-details">
              <summary>Technical Details</summary>
              <pre>{JSON.stringify(error.details, null, 2)}</pre>
            </details>
          )}
        </div>
        <div className="modal-footer">
          <button className="button-secondary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

// Confirmation Modal Component
const ConfirmationModal = ({ isOpen, onClose, onConfirm, title, message, confirmText = "Delete", isDestructive = true }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>{title}</h3>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          <div className="confirmation-icon">
            {isDestructive ? '🗑️' : '❓'}
          </div>
          <div className="confirmation-message">
            {message}
          </div>
        </div>
        <div className="modal-footer">
          <button className="button-secondary" onClick={onClose}>
            Cancel
          </button>
          <button 
            className={isDestructive ? "button-destructive" : "button-primary"} 
            onClick={onConfirm}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
};

// Success Toast Component
const Toast = ({ message, type = 'success', isVisible, onClose }) => {
  useEffect(() => {
    if (isVisible) {
      const timer = setTimeout(onClose, 4000);
      return () => clearTimeout(timer);
    }
  }, [isVisible, onClose]);

  if (!isVisible) return null;

  return (
    <div className={`toast toast-${type}`}>
      <span className="toast-icon">
        {type === 'success' ? '✅' : type === 'warning' ? '⚠️' : 'ℹ️'}
      </span>
      <span className="toast-message">{message}</span>
      <button className="toast-close" onClick={onClose}>&times;</button>
    </div>
  );
};

const UploadDocument = () => {
  const [activeTab, setActiveTab] = useState('files');
  const [files, setFiles] = useState([]);
  const [urls, setUrls] = useState('');
  const [singleUrl, setSingleUrl] = useState('');
  const [urlInputMethod, setUrlInputMethod] = useState('single');
  const [uploading, setUploading] = useState(false);
  const [uploadResults, setUploadResults] = useState([]);
  const [documentStatus, setDocumentStatus] = useState(null);
  const [supportedTypes, setSupportedTypes] = useState({});
  
  // New state for enhanced UI
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState('');
  const [error, setError] = useState(null);
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [toast, setToast] = useState({ message: '', type: 'success', visible: false });

  // Delete functionality state
  const [selectedDocuments, setSelectedDocuments] = useState(new Set());
  const [isDeleting, setIsDeleting] = useState(false);
  const [showDeleteConfirmation, setShowDeleteConfirmation] = useState(false);
  const [deleteType, setDeleteType] = useState('selected'); // 'selected' or 'all'
  const [selectAll, setSelectAll] = useState(false);

  const API_BASE_URL = 'http://localhost:8000';

  useEffect(() => {
    fetchSupportedTypes();
    fetchDocumentStatus();
  }, []);

  // Update selectAll state when documents change
  useEffect(() => {
    if (documentStatus?.documents) {
      const allDocIds = Object.keys(documentStatus.documents);
      setSelectAll(allDocIds.length > 0 && allDocIds.every(id => selectedDocuments.has(id)));
    }
  }, [selectedDocuments, documentStatus]);

  const showToast = (message, type = 'success') => {
    setToast({ message, type, visible: true });
  };

  const hideToast = () => {
    setToast(prev => ({ ...prev, visible: false }));
  };

  const showError = (error, title = "Upload Error") => {
    setError({ ...error, title });
    setShowErrorModal(true);
  };

  const hideError = () => {
    setShowErrorModal(false);
    setError(null);
  };

  const fetchSupportedTypes = async () => {
    try {
      setProgressLabel('Loading supported file types...');
      const response = await fetch(`${API_BASE_URL}/supported-types`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch supported types: ${response.statusText}`);
      }
      
      const data = await response.json();
      setSupportedTypes(data.supported_types);
    } catch (error) {
      console.error('Error fetching supported types:', error);
      showError({
        message: 'Failed to load supported file types',
        details: { error: error.message }
      }, 'Configuration Error');
    }
  };

  const fetchDocumentStatus = async () => {
    try {
      setProgressLabel('Loading document status...');
      const response = await fetch(`${API_BASE_URL}/documents`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch document status: ${response.statusText}`);
      }
      
      const data = await response.json();
      setDocumentStatus(data);
    } catch (error) {
      console.error('Error fetching document status:', error);
      showToast('Could not load document status', 'warning');
    }
  };

  const handleFileChange = (event) => {
    const selectedFiles = Array.from(event.target.files);
    setFiles(selectedFiles);
    
    // Validate file types
    const invalidFiles = selectedFiles.filter(file => {
      const extension = '.' + file.name.split('.').pop().toLowerCase();
      return !supportedTypes[extension];
    });
    
    if (invalidFiles.length > 0) {
      showError({
        message: `Unsupported file types detected`,
        details: { 
          invalidFiles: invalidFiles.map(f => f.name),
          supportedTypes: Object.keys(supportedTypes)
        }
      }, 'File Type Error');
    }
  };

  const handleFileUpload = async () => {
    if (files.length === 0) {
      showError({ 
        message: 'Please select files to upload',
        details: { action: 'Select at least one file before uploading' }
      }, 'No Files Selected');
      return;
    }

    setUploading(true);
    setUploadResults([]);
    setProgress(0);
    setProgressLabel(`Uploading ${files.length} file(s)...`);

    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    try {
      // Simulate progress for demo (in real app, you'd get this from upload progress)
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      const response = await fetch(`${API_BASE_URL}/upload-files`, {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setProgress(100);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed (${response.status}): ${errorText}`);
      }

      const results = await response.json();
      setUploadResults(results);
      
      // Check for any failed uploads
      const failedUploads = results.filter(r => !r.success);
      const successfulUploads = results.filter(r => r.success);
      
      if (successfulUploads.length > 0) {
        showToast(`Successfully uploaded ${successfulUploads.length} file(s)`, 'success');
      }
      
      if (failedUploads.length > 0) {
        showError({
          message: `${failedUploads.length} file(s) failed to upload`,
          details: { failedUploads: failedUploads.map(f => f.message) }
        }, 'Partial Upload Failure');
      }
      
      fetchDocumentStatus();
    } catch (error) {
      console.error('Error uploading files:', error);
      showError({
        message: 'Failed to upload files',
        details: { 
          error: error.message,
          timestamp: new Date().toISOString()
        }
      }, 'Upload Error');
      
      setUploadResults([{
        success: false,
        message: `Error uploading files: ${error.message}`
      }]);
    } finally {
      setUploading(false);
      setProgress(0);
      setProgressLabel('');
    }
  };

  const handleUrlUpload = async () => {
    const urlList = urlInputMethod === 'single' 
      ? [singleUrl].filter(url => url.trim()) 
      : urls.split('\n').filter(url => url.trim());

    if (urlList.length === 0) {
      showError({
        message: 'Please enter URLs to process',
        details: { action: 'Enter at least one valid URL' }
      }, 'No URLs Provided');
      return;
    }

    // Validate URLs
    const invalidUrls = urlList.filter(url => {
      try {
        new URL(url);
        return false;
      } catch {
        return true;
      }
    });

    if (invalidUrls.length > 0) {
      showError({
        message: 'Invalid URL format detected',
        details: { invalidUrls }
      }, 'URL Validation Error');
      return;
    }

    setUploading(true);
    setUploadResults([]);
    setProgress(0);
    setProgressLabel(`Processing ${urlList.length} URL(s)...`);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 15;
        });
      }, 300);

      const response = await fetch(`${API_BASE_URL}/upload-urls`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ urls: urlList }),
      });

      clearInterval(progressInterval);
      setProgress(100);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`URL processing failed (${response.status}): ${errorText}`);
      }

      const results = await response.json();
      setUploadResults(results);
      
      const failedProcessing = results.filter(r => !r.success);
      const successfulProcessing = results.filter(r => r.success);
      
      if (successfulProcessing.length > 0) {
        showToast(`Successfully processed ${successfulProcessing.length} URL(s)`, 'success');
      }
      
      if (failedProcessing.length > 0) {
        showError({
          message: `${failedProcessing.length} URL(s) failed to process`,
          details: { failedUrls: failedProcessing.map(f => f.message) }
        }, 'URL Processing Error');
      }
      
      fetchDocumentStatus();
    } catch (error) {
      console.error('Error uploading URLs:', error);
      showError({
        message: 'Failed to process URLs',
        details: { 
          error: error.message,
          urls: urlList,
          timestamp: new Date().toISOString()
        }
      }, 'URL Processing Error');
      
      setUploadResults([{
        success: false,
        message: `Error processing URLs: ${error.message}`
      }]);
    } finally {
      setUploading(false);
      setProgress(0);
      setProgressLabel('');
    }
  };

  // Document selection handlers
  const handleDocumentSelect = (indexName, isSelected) => {
    const newSelected = new Set(selectedDocuments);
    if (isSelected) {
      newSelected.add(indexName);
    } else {
      newSelected.delete(indexName);
    }
    setSelectedDocuments(newSelected);
  };

  const handleSelectAll = (isSelected) => {
    if (isSelected && documentStatus?.documents) {
      setSelectedDocuments(new Set(Object.keys(documentStatus.documents)));
    } else {
      setSelectedDocuments(new Set());
    }
    setSelectAll(isSelected);
  };

  // Delete handlers
  const handleDeleteSelected = () => {
    if (selectedDocuments.size === 0) {
      showToast('No documents selected for deletion', 'warning');
      return;
    }
    setDeleteType('selected');
    setShowDeleteConfirmation(true);
  };

  const handleDeleteAll = () => {
    if (!documentStatus?.documents || Object.keys(documentStatus.documents).length === 0) {
      showToast('No documents to delete', 'warning');
      return;
    }
    setDeleteType('all');
    setShowDeleteConfirmation(true);
  };

  const confirmDelete = async () => {
    setShowDeleteConfirmation(false);
    setIsDeleting(true);
    setProgressLabel(deleteType === 'all' ? 'Deleting all documents...' : `Deleting ${selectedDocuments.size} document(s)...`);

    try {
      if (deleteType === 'all') {
        const response = await fetch(`${API_BASE_URL}/documents`, {
          method: 'DELETE',
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Failed to delete all documents: ${errorText}`);
        }

        const result = await response.json();
        if (result.success) {
          showToast('All documents deleted successfully', 'success');
          setSelectedDocuments(new Set());
          setSelectAll(false);
        } else {
          throw new Error(result.message);
        }
      } else {
        // Delete selected documents
        const deletePromises = Array.from(selectedDocuments).map(async (indexName) => {
          const response = await fetch(`${API_BASE_URL}/documents/${indexName}`, {
            method: 'DELETE',
          });
          
          if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Failed to delete ${indexName}: ${errorText}`);
          }
          
          return response.json();
        });

        const results = await Promise.allSettled(deletePromises);
        const successful = results.filter(r => r.status === 'fulfilled' && r.value.success).length;
        const failed = results.filter(r => r.status === 'rejected' || !r.value.success).length;

        if (successful > 0) {
          showToast(`Successfully deleted ${successful} document(s)`, 'success');
          setSelectedDocuments(new Set());
          setSelectAll(false);
        }

        if (failed > 0) {
          showError({
            message: `Failed to delete ${failed} document(s)`,
            details: { 
              failedDeletions: results
                .filter(r => r.status === 'rejected' || !r.value.success)
                .map(r => r.reason?.message || r.value?.message || 'Unknown error')
            }
          }, 'Delete Error');
        }
      }

      fetchDocumentStatus();
    } catch (error) {
      console.error('Error deleting documents:', error);
      showError({
        message: 'Failed to delete documents',
        details: { error: error.message }
      }, 'Delete Error');
    } finally {
      setIsDeleting(false);
      setProgressLabel('');
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getDeleteConfirmationMessage = () => {
    if (deleteType === 'all') {
      return `Are you sure you want to delete ALL documents? This will permanently remove ${documentStatus?.total_docs || 0} document(s) and their associated indices.`;
    } else {
      return `Are you sure you want to delete ${selectedDocuments.size} selected document(s)? This action cannot be undone.`;
    }
  };

  return (
    <div className="upload-container">
      <div className="header">
        <h1>📚 Universal Document Upload System</h1>
        <p>Upload various file types or provide URLs to process documents into searchable indices.</p>
      </div>

      <div className="main-content">
        <div className="upload-section">
          <div className="tabs">
            <button 
              className={`tab ${activeTab === 'files' ? 'active' : ''}`}
              onClick={() => setActiveTab('files')}
            >
              📁 File Upload
            </button>
            <button 
              className={`tab ${activeTab === 'urls' ? 'active' : ''}`}
              onClick={() => setActiveTab('urls')}
            >
              🌐 URL Upload
            </button>
          </div>

          <div className="tab-content">
            {activeTab === 'files' && (
              <div className="file-upload-tab">
                <h3>Upload Files</h3>
                <p className="supported-types">
                  <strong>Supported file types:</strong> {Object.values(supportedTypes).join(', ')}
                </p>
                
                <div className="file-input-container">
                  <input
                    type="file"
                    multiple
                    onChange={handleFileChange}
                    className="file-input"
                    id="file-upload"
                    accept={Object.keys(supportedTypes).map(ext => ext.substring(1)).join(',')}
                    disabled={uploading}
                  />
                  <label htmlFor="file-upload" className="file-input-label">
                    Choose Files
                  </label>
                </div>

                {files.length > 0 && (
                  <div className="selected-files">
                    <h4>Selected Files:</h4>
                    <ul>
                      {files.map((file, index) => (
                        <li key={index} className="file-item">
                          <span className="file-name">{file.name}</span>
                          <span className="file-size">({formatFileSize(file.size)})</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <button
                  onClick={handleFileUpload}
                  disabled={uploading || files.length === 0}
                  className="upload-button"
                >
                  {uploading ? 'Processing...' : 'Upload Files'}
                </button>
              </div>
            )}

            {activeTab === 'urls' && (
              <div className="url-upload-tab">
                <h3>Upload from URLs</h3>
                <p>Enter URLs to web pages, documents, or files to download and process.</p>

                <div className="url-input-method">
                  <label>
                    <input
                      type="radio"
                      value="single"
                      checked={urlInputMethod === 'single'}
                      onChange={(e) => setUrlInputMethod(e.target.value)}
                      disabled={uploading}
                    />
                    Single URL
                  </label>
                  <label>
                    <input
                      type="radio"
                      value="multiple"
                      checked={urlInputMethod === 'multiple'}
                      onChange={(e) => setUrlInputMethod(e.target.value)}
                      disabled={uploading}
                    />
                    Multiple URLs (one per line)
                  </label>
                </div>

                {urlInputMethod === 'single' ? (
                  <input
                    type="url"
                    value={singleUrl}
                    onChange={(e) => setSingleUrl(e.target.value)}
                    placeholder="https://example.com/document.pdf"
                    className="url-input"
                    disabled={uploading}
                  />
                ) : (
                  <textarea
                    value={urls}
                    onChange={(e) => setUrls(e.target.value)}
                    placeholder="https://example.com/doc1.pdf&#10;https://example.com/doc2.html&#10;https://example.com/page3"
                    className="url-textarea"
                    rows={5}
                    disabled={uploading}
                  />
                )}

                <button
                  onClick={handleUrlUpload}
                  disabled={uploading || (urlInputMethod === 'single' ? !singleUrl.trim() : !urls.trim())}
                  className="upload-button"
                >
                  {uploading ? 'Processing...' : 'Process URLs'}
                </button>
              </div>
            )}
          </div>

          {(uploading || isDeleting) && (
            <div className="loading-container">
              <ProgressBar 
                isIndeterminate={progress === 0} 
                progress={progress}
                label={progressLabel}
              />
            </div>
          )}

          {uploadResults.length > 0 && (
            <div className="results-container">
              <h3>Upload Results</h3>
              {uploadResults.map((result, index) => (
                <div key={index} className={`result-item ${result.success ? 'success' : 'error'}`}>
                  <div className="result-icon">
                    {result.success ? '✅' : '❌'}
                  </div>
                  <div className="result-content">
                    <p className="result-message">{result.message}</p>
                    {result.success && result.index_name && (
                      <div className="result-details">
                        <small>Index: {result.index_name} | Chunks: {result.chunks}</small>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="sidebar">
          <div className="system-info">
            <h3>📚 Document Upload System</h3>
            <div className="features">
              <h4>Features</h4>
              <ul>
                <li><strong>Multi-format support:</strong> PDF, DOCX, TXT, CSV, Excel, HTML, JSON, XML, Markdown</li>
                <li><strong>URL processing:</strong> Download and process web content</li>
                <li><strong>FAISS indexing:</strong> Production-ready vector storage</li>
                <li><strong>Metadata tracking:</strong> Complete document management</li>
                <li><strong>Chunk optimization:</strong> Smart text splitting for better retrieval</li>
                <li><strong>Document management:</strong> Select and delete documents</li>
              </ul>
            </div>
            {Object.keys(supportedTypes).length > 0 && (
              <div className="supported-formats">
                <h4>Supported Formats</h4>
                <ul>
                  {Object.entries(supportedTypes).map(([ext, desc]) => (
                    <li key={ext}><strong>{ext}</strong>: {desc}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>

      {documentStatus && (
        <div className="document-status">
          <h3>📊 Document Library Status</h3>
          
          {documentStatus.total_docs === 0 ? (
            <div className="no-documents">
              <p>No documents uploaded yet. Upload some documents to get started!</p>
            </div>
          ) : (
            <>
              <div className="status-metrics">
                <div className="metric">
                  <h4>{documentStatus.total_docs}</h4>
                  <p>Total Documents</p>
                </div>
                <div className="metric">
                  <h4>{documentStatus.total_chunks}</h4>
                  <p>Total Chunks</p>
                </div>
                <div className="metric">
                  <h4>{Object.keys(documentStatus.file_types).length}</h4>
                  <p>File Types</p>
                </div>
              </div>

              {/* Document Management Controls */}
              <div className="document-management">
                <div className="management-controls">
                  <div className="select-controls">
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={selectAll}
                        onChange={(e) => handleSelectAll(e.target.checked)}
                        disabled={isDeleting}
                      />
                      Select All ({Object.keys(documentStatus.documents).length})
                    </label>
                    {selectedDocuments.size > 0 && (
                      <span className="selected-count">
                        {selectedDocuments.size} selected
                      </span>
                    )}
                  </div>
                  <div className="action-controls">
                    <button
                      onClick={handleDeleteSelected}
                      disabled={selectedDocuments.size === 0 || isDeleting}
                      className="button-destructive"
                    >
                      🗑️ Delete Selected ({selectedDocuments.size})
                    </button>
                    <button
                      onClick={handleDeleteAll}
                      disabled={Object.keys(documentStatus.documents).length === 0 || isDeleting}
                      className="button-destructive"
                    >
                      🗑️ Delete All
                    </button>
                  </div>
                </div>
              </div>

              {Object.keys(documentStatus.file_types).length > 0 && (
                <div className="file-type-distribution">
                  <h4>File Type Distribution:</h4>
                  <ul>
                    {Object.entries(documentStatus.file_types).map(([type, count]) => (
                      <li key={type}>{type}: {count} document(s)</li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="document-list">
                <details open>
                  <summary>Document Library ({Object.keys(documentStatus.documents).length})</summary>
                  <div className="document-items">
                    {Object.entries(documentStatus.documents).map(([indexName, docInfo]) => (
                      <div key={indexName} className="document-item">
                        <div className="document-header">
                          <label className="document-checkbox">
                            <input
                              type="checkbox"
                              checked={selectedDocuments.has(indexName)}
                              onChange={(e) => handleDocumentSelect(indexName, e.target.checked)}
                              disabled={isDeleting}
                            />
                          </label>
                          <h5 className="document-title">{docInfo.original_name}</h5>
                        </div>
                        <div className="document-details">
                          <div className="detail-row">
                            <span>Type: {docInfo.file_type || 'N/A'}</span>
                            <span>Pages: {docInfo.pages || 'N/A'}</span>
                            <span>Chunks: {docInfo.chunks || 'N/A'}</span>
                            <span>Source: {docInfo.source_type || 'N/A'}</span>
                          </div>
                          {docInfo.source_url && (
                            <div className="detail-row">
                              <span>URL: <a href={docInfo.source_url} target="_blank" rel="noopener noreferrer">{docInfo.source_url}</a></span>
                            </div>
                          )}
                          <div className="detail-row">
                            <span>Created: {new Date(docInfo.created_at).toLocaleString()}</span>
                            <span>Index: <code>{indexName}</code></span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </details>
              </div>
            </>
          )}
        </div>
      )}

      {/* Error Modal */}
      <ErrorModal 
        isOpen={showErrorModal}
        onClose={hideError}
        error={error}
        title={error?.title}
      />

      {/* Delete Confirmation Modal */}
      <ConfirmationModal
        isOpen={showDeleteConfirmation}
        onClose={() => setShowDeleteConfirmation(false)}
        onConfirm={confirmDelete}
        title={deleteType === 'all' ? 'Delete All Documents' : 'Delete Selected Documents'}
        message={getDeleteConfirmationMessage()}
        confirmText="Delete"
        isDestructive={true}
      />

      {/* Toast Notifications */}
      <Toast 
        message={toast.message}
        type={toast.type}
        isVisible={toast.visible}
        onClose={hideToast}
      />
    </div>
  );
};

export default UploadDocument;