import { useEffect, useMemo, useState } from "react";
import { JsonFile, JsonFilePrompt } from "./Comparer";
import { Prompt } from "./components/Prompt";
import './FileComparer.css'
import { useStateWithLocalStorageBoolean, useStateWithLocalStorageInt, useStateWithLocalStorageString } from "./useHooks";

export interface PromptResults {
  file: JsonFile;
  results: JsonFilePrompt[];
}

export const FileComparer = ({files}: {files:JsonFile[]}) => {
  const samplingMethods = useMemo(() => {
    const s = new Set<string>();
    files.forEach(file => file?.prompts?.forEach(p => p.results.forEach(result => s.add(result.sampling_config))));
    return Array.from(s.values());
  }, [files]);

  const [samplingMethod, setSamplingMethod] = useStateWithLocalStorageString(samplingMethods[0] || 'beam5', 'samplingMethod');
  const [outputIndex, setOutputIndex] = useStateWithLocalStorageInt(0, 'outputIndex'); // -1 for all
  const [showSamplingConfig, setShowSamplingConfig] = useStateWithLocalStorageBoolean(true, 'samplingConfig');
  const [renderMarkdown, setRenderMarkdown] = useStateWithLocalStorageBoolean(true, 'renderMarkdown');
  const [expandedPrompts, setExpandedPrompts] = useState<Set<string>>(new Set<string>());

  useEffect(() => {
    if (samplingMethods && samplingMethods.length > 0 && samplingMethod && !samplingMethods.includes(samplingMethod)) {
      setSamplingMethod(samplingMethods[0] || '');
    }
  }, [samplingMethods, setSamplingMethod, samplingMethod]);

  const toggleExpandedPrompts = () => {
    if(expandedPrompts.size === 0) {
      setExpandedPrompts(new Set(Object.keys(prompts)));
    } else {
      setExpandedPrompts(new Set());
    }
  }
  const toggleCollapsed = (prompt: string, value: boolean) => {
    const newExpandedPrompts = new Set(expandedPrompts);
    if (value) {
      newExpandedPrompts.add(prompt);
    } else {
      newExpandedPrompts.delete(prompt);
    }
    setExpandedPrompts(newExpandedPrompts);
  }

  const prompts = useMemo(() => {
    const prompts: {[prompt: string]: PromptResults[]} = {};
    files.forEach(file => {
      file.prompts?.forEach(p => {
        if (!prompts[p.prompt]) {
          prompts[p.prompt] = [];
        }
        prompts[p.prompt] = [...prompts[p.prompt], {file, results: p.results.filter(r => !samplingMethod || (r.sampling_config === samplingMethod))}];
      });
    });
    return prompts;
  }, [files, samplingMethod]);

  return (
    <div>
      <div className="config_options">
        <div>
          <label htmlFor="samplingMethod">Sampling Method: &nbsp;</label>
          <select id="samplingMethod" value={samplingMethod} onChange={(e) => setSamplingMethod(e.target.value)}>
            <option value="">Show all</option>
            {samplingMethods.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div>
          <label htmlFor="outputIndex">Output Index: &nbsp;</label>
          <select id="outputIndex" value={outputIndex} onChange={(e) => setOutputIndex(parseInt(e.target.value))}>
            <option value={-1}>Show all</option>
            <option value={0}>0</option>
            <option value={1}>1</option>
          </select>
        </div>
        <div className="configContainer">
          <input type="checkbox" id="show_sampling_config" checked={showSamplingConfig} onChange={(e) => setShowSamplingConfig(e.target.checked)}/>
          <label htmlFor="show_sampling_config">Show&nbsp;params</label>
        </div>
        <div className="configContainer">
          <input type="checkbox" id="render_markdown" checked={renderMarkdown} onChange={(e) => setRenderMarkdown(e.target.checked)}/>
          <label htmlFor="render_markdown">Render&nbsp;Markdown</label>
        </div>
        <div>
          <button onClick={toggleExpandedPrompts}>{expandedPrompts.size === 0 ? 'Expand all' : 'Collapse all'}</button>
        </div>
      </div>
      {Object.keys(prompts).map((p) =>
        <Prompt
          key={p}
          prompt={p}
          results={prompts[p]}
          outputIndex={outputIndex}
          showSamplingMethod={samplingMethod === ''}
          showSamplingConfig={showSamplingConfig}
          collapsed={!expandedPrompts.has(p)}
          onToggleCollapsed={() => toggleCollapsed(p, !expandedPrompts.has(p))}
          renderMarkdown={renderMarkdown}
          />
      )}
    </div>
  );
}
