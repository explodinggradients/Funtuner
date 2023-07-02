import React, { useMemo, useState } from 'react';
import './Comparer.css';
import { FileComparer } from './FileComparer';
import { FileLoaderTextArea } from './FileLoaderTextArea';

export interface JsonFilePrompt {
  outputs: string[];
  sampling_config: string;
  sampling_params: {[key: string]: string | number | boolean};
}

export interface JsonFile {
  model_name: string;
  date: string;
  args: {[key: string]: any};
  prompts: {prompt: string, results: JsonFilePrompt[]}[];
}

export const Comparer = () => {
  const [files, setFiles] = useState<(JsonFile | undefined)[]>([]); // Can be undefined when loading happens out of order
  const [localFiles, setLocalFiles] = useState<JsonFile[]>([]);

  function localFileAdded(file: JsonFile) {
    setLocalFiles(localFiles => [...localFiles, file]);
  }

  const filesMerged = useMemo(() => ([...files, ...localFiles].filter( f => f) as JsonFile[]), [files, localFiles]);

  return (
    <div className="comparer">
      <h1>Battle of LLMs</h1>
      <FileLoaderTextArea setFiles={setFiles} localFileAdded={localFileAdded} localFiles={localFiles} />
      <FileComparer files={filesMerged} />
    </div>
  );
}
