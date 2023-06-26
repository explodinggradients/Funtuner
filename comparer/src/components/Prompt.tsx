import { PromptResults } from "../FileComparer";
import ReactMarkdown from 'react-markdown';
import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import {dark} from 'react-syntax-highlighter/dist/esm/styles/prism'
import './Prompt.css';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css'

interface PromptProps {
  prompt: string;
  results: PromptResults[];
  showSamplingMethod: boolean;
  outputIndex: number;
  showSamplingConfig: boolean;
  collapsed?: boolean;
  onToggleCollapsed?: () => void;
  renderMarkdown: boolean;
};

export const Prompt = ({collapsed, onToggleCollapsed, prompt, results, outputIndex, showSamplingMethod, showSamplingConfig, renderMarkdown}: PromptProps) => {
  return (
    <div>
      <div className="promptBubble" onClick={onToggleCollapsed}><RenderMarkdown renderMarkdown={renderMarkdown}>{prompt}</RenderMarkdown></div>
      { !collapsed &&
        <table className={showSamplingMethod ? 'replyTableWithAltRows' : ''}>
          <tbody>
            {results.map((result, modelIndex) => <tr key={result.file.model_name}>
              <td className="model_name">{result.file.model_name}</td>
              <td>{result.results.map((result, result_index) => (
                <div key={result_index}>
                  {(showSamplingMethod || showSamplingConfig) && <div className="sampling_config">
                    {showSamplingMethod && <div>Sampling config: <b>{result.sampling_config}</b></div>}
                    {showSamplingConfig && Object.keys(result.sampling_params).map(param => <div key={param}>{param}: <span className="param-value">{result.sampling_params[param].toString()}</span></div>)}
                  </div>}
                  {outputIndex === -1 ?
                    result.outputs.map((output, index) => <ReplyBubble key={index} modelIndex={modelIndex} output={output} saturation={result_index} renderMarkdown={renderMarkdown}/>) :
                    <ReplyBubble modelIndex={modelIndex} output={result.outputs[outputIndex]} saturation={result_index} renderMarkdown={renderMarkdown} />
                  }
                </div>
              ))}</td>
            </tr>)}
          </tbody>
        </table>
      }
    </div>
  );
}

type ReplyBubbleProps = { modelIndex: number, output: string, saturation: number, renderMarkdown: boolean };

const endoftextToken = '<|endoftext|>';
const ReplyBubble = ({modelIndex, output, saturation, renderMarkdown} : ReplyBubbleProps) => {
  // remove <|endoftext|> from the end of the output if it is present
  if (output.endsWith(endoftextToken)) {
    output = output.slice(0, -endoftextToken.length);
  }
  const out = <RenderMarkdown renderMarkdown={renderMarkdown}>{output}</RenderMarkdown>;
  return <div className="replyBubble" style={{background: `hsl(${194 + 42.3 * modelIndex} ${30 + 23 / (saturation + 1)}% ${73 + 10 / (saturation+1)}%)`}}>{out}</div>
}

type RenderMarkdownProps = {
  children: string;
  renderMarkdown: boolean
}

const RenderMarkdown = ({children, renderMarkdown} : RenderMarkdownProps ) => {
  return renderMarkdown ? <ReactMarkdown
  children={children}
  remarkPlugins={[remarkMath]}
  rehypePlugins={[rehypeKatex]}
  components={{
    code({node, inline, className, children, ...props}) {
      const match = /language-(\w+)/.exec(className || '')
      return !inline && match ? (
        <SyntaxHighlighter
          children={String(children).replace(/\n$/, '')}
          style={dark as any}
          language={match[1]}
          PreTag="div"
          {...props}
        />
      ) : (
        <code className={className} {...props}>
          {children}
        </code>
      )
    }
  }}
  /> : <div className="noMarkdown">{children}</div>;
}