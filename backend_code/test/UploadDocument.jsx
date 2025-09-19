import React, { useState, useEffect } from 'react';
import './UploadDocument.css';

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

  const API_BASE_URL = 'http://localhost:8000';

  // Fetch supported file types on component mount
  useEffect(() => {
    fetchSupportedTypes();
    fetchDocumentStatus();
  }, []);

  const fetchSupportedTypes = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/supported-types`);
      const data = await response.json();
      setSupportedTypes(data.supported_types);
    } catch (error) {
      console.error('Error fetching supported types:', error);
    }
  };

  const fetchDocumentStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents`);
      const data = await response.json();
      setDocumentStatus(data);
    } catch (error) {
      console.error('Error fetching document status:', error);
    }
  };

  const handleFileChange = (event) => {
    const selectedFiles = Array.from(event.target.files);
    setFiles(selectedFiles);
  };

  const handleFileUpload = async () => {
    if (files.length === 0) {
      alert('Please select files to upload');
      return;
    }

    setUploading(true);
    setUploadResults([]);

    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    try {
      const response = await fetch(`${API_BASE_URL}/upload-files`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const results = await response.json();
      setUploadResults(results);
      fetchDocumentStatus(); // Refresh document status
    } catch (error) {
      console.error('Error uploading files:', error);
      setUploadResults([{
        success: false,
        message: `Error uploading files: ${error.message}`
      }]);
    } finally {
      setUploading(false);
    }
  };

  const handleUrlUpload = async () => {
    const urlList = urlInputMethod === 'single' 
      ? [singleUrl].filter(url => url.trim()) 
      : urls.split('\n').filter(url => url.trim());

    if (urlList.length === 0) {
      alert('Please enter URLs to process');
      return;
    }

    setUploading(true);
    setUploadResults([]);

    try {
      const response = await fetch(`${API_BASE_URL}/upload-urls`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ urls: urlList }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const results = await response.json();
      setUploadResults(results);
      fetchDocumentStatus(); // Refresh document status
    } catch (error) {
      console.error('Error uploading URLs:', error);
      setUploadResults([{
        success: false,
        message: `Error processing URLs: ${error.message}`
      }]);
    } finally {
      setUploading(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
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
                    />
                    Single URL
                  </label>
                  <label>
                    <input
                      type="radio"
                      value="multiple"
                      checked={urlInputMethod === 'multiple'}
                      onChange={(e) => setUrlInputMethod(e.target.value)}
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
                  />
                ) : (
                  <textarea
                    value={urls}
                    onChange={(e) => setUrls(e.target.value)}
                    placeholder="https://example.com/doc1.pdf&#10;https://example.com/doc2.html&#10;https://example.com/page3"
                    className="url-textarea"
                    rows={5}
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

          {uploading && (
            <div className="loading-container">
              <div className="loading-spinner"></div>
              <p>Processing documents...</p>
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
                <details>
                  <summary>View All Documents</summary>
                  <div className="document-items">
                    {Object.entries(documentStatus.documents).map(([indexName, docInfo]) => (
                      <div key={indexName} className="document-item">
                        <h5>{docInfo.original_name}</h5>
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
    </div>
  );
};

export default UploadDocument;