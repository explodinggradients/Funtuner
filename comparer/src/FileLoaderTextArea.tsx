import { useCallback, useEffect, useState } from "react";
import { JsonFile } from "./Comparer";
import { useDropzone } from 'react-dropzone';
import './FileLoaderTextArea.css';

const someUrls = `https://raw.githubusercontent.com/LAION-AI/Open-Assistant/main/model/model_eval/manual/sampling_reports/2023-03-01_theblackcat102_pythia-12b-deduped-sft_sampling.json
https://raw.githubusercontent.com/LAION-AI/Open-Assistant/main/model/model_eval/manual/sampling_reports/2023-03-01_theblackcat102_pythia-1b-deduped-sft_sampling.json
https://raw.githubusercontent.com/LAION-AI/Open-Assistant/main/model/model_eval/manual/sampling_reports/2023-03-01_theblackcat102_pythia-3b-deduped-sft_sampling.json`;

const fileCache: {[key:string]:JsonFile } = {}

async function catchedFetch(url: string) {
  if (fileCache[url]) {
    return fileCache[url];
  }
  return await fetch(url).then(r => r.json()).then(json => {
    fileCache[url] = json;
    return json;
  });
}

interface FileLoaderTextAreaProps {
  setFiles: (files: (JsonFile| undefined)[] | ((f: (JsonFile | undefined)[]) => (JsonFile| undefined)[])) => void;
  localFileAdded: (file: JsonFile) => void;
  localFiles: JsonFile[];
}

export const FileLoaderTextArea = ({setFiles, localFiles, localFileAdded} : FileLoaderTextAreaProps) => {
  // get window location param for ?f=
  const urlParams = new URLSearchParams(window.location.search);
  const filename = urlParams.get('f');
  const [filenamesTxt, setFilenamesTxt] = useState<string>(filename ?? localStorage.getItem('filenames') ?? someUrls);
  const [fileErrors, setFileErrors] = useState<string[]>([]);
  const [loading, setLoading] = useState<number>(0);

  useEffect(() => {
    localStorage.setItem('filenames', filenamesTxt);
    // Update window location search param with f= filenamesTxt
    const url = new URL(window.location.href);
    if (filenamesTxt) {
      url.searchParams.set('f', filenamesTxt);
    } else {
      url.searchParams.delete('f');
    }
    window.history.pushState({}, '', url.toString());
  }, [filenamesTxt]);

  useEffect(() => {
    const urls = filenamesTxt.split(/[\r\n]/).map(s => s.trim()).filter(s => s);
    setLoading(urls.length);
    setFileErrors([]);
    setFiles([]);
    urls.forEach((url, index) => catchedFetch(url).then(json => {
      setFiles(fs => {
        const fs_ = [...fs];
        fs_[index] = json;
        return fs_;
      });
      setFileErrors(es => { const es_ = [...es]; es_[index] = ''; return es_});
    }).catch(e => {
      setFileErrors(es => { const es_ = [...es]; es_[index] = `Failed to load ${url}: ${e.toString()}`; return es_});
    }).finally(() => setLoading(l => l - 1)));
  }, [filenamesTxt, setFiles]);

  return (
    <>
      <FileDropzone localFileAdded={localFileAdded} localFiles={localFiles} loadingComponent={<>
          {loading > 0 && <div className="loading">Loading... ({loading} more)</div>}
          <div className="errors">{fileErrors.map((e, i) => e && <div className="error" key={i}>{e}</div>)}</div></>
        }>
        <div className={loading > 0 ? "loading_wait" : fileErrors.some(e => e) ? "loading_errors" : "loading_success"}>
          <textarea className="filenames" placeholder="Enter url to json file or drag and drop files in" value={filenamesTxt} onChange={(e) => setFilenamesTxt(e.target.value)} />
        </div>
      </FileDropzone>
    </>
  );
}

interface FileDropzoneProps {
  children?: React.ReactNode;
  loadingComponent?: React.ReactNode;
  localFileAdded: (file: JsonFile) => void;
  localFiles: JsonFile[];
}

function FileDropzone({children, loadingComponent, localFileAdded, localFiles}: FileDropzoneProps) {
  const [loading, setLoading] = useState(0);
  const [error, setError] = useState('');
  const onDrop = useCallback((acceptedFiles: File[]) => {
    setLoading(acceptedFiles.length);
    setError('');
    acceptedFiles.forEach((file) => {
      const reader = new FileReader();
      reader.onabort = () => { setError('file reading was aborted'); setLoading(l => l - 1); };
      reader.onerror = () => { setError('file reading has failed'); setLoading(l => l - 1); };
      reader.onload = () => {
        setLoading(l => l - 1);
        try {
          localFileAdded(JSON.parse(reader.result as string));
        } catch(e: any) {
          setError(`Failed to load ${file.name}: ${e.toString()}`);
        }
      }
      reader.readAsText(file);
    })
  }, [localFileAdded]);

  const {getRootProps, getInputProps} = useDropzone({onDrop})

  return (
    <div {...getRootProps()} onClick={() => null}>
      {children}
      {loading > 0 && <div className="loading">Loading... ({loading} more)</div>}
      <div className="dropzone" >
        {loadingComponent}
       <div onClick={getRootProps().onClick} className="dropzone_messages">
          <input {...getInputProps()} />
          {<div className="local_files">{localFiles.map((f, i) => <div key={i}>{f.model_name}</div>)}</div>}
          <div className="hint">Drag 'n' drop some files here, or click to select files</div>
        </div>
        <div className="loading_errors" style={{fontSize: 12}}>{error}</div>
      </div>
    </div>
  )
}
