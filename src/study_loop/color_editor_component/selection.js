/* Pure selection transformation: no network, no evaluation or HTML insertion. */
(function (root) {
  "use strict";
  const open = /^<span\s+style=["']\s*color\s*:\s*#(222222|D62828|2563EB)\s*;?\s*["']\s*>$/i;
  function ranges(text, expression) { return Array.from(text.matchAll(expression), m => [m.index, m.index+m[0].length]); }
  const codePattern = /(^[ \t]*(`{3,}|~{3,})[^\n]*\n[\s\S]*?^[ \t]*\2[ \t]*$)|(`+)[^\n]*?\3/gm;
  const mathPattern = /\$\$[\s\S]*?\$\$|(?<!\\)\$[^$\n]+?\$|\\\[[\s\S]*?\\\]|\\\([^\n]*?\\\)/g;
  const inside = (index, rs) => rs.some(([a,b]) => a<=index && index<b);
  function applyColor(text, start, end, color) {
    if (!['222222','D62828','2563EB'].includes(color)) throw Error('请选择黑色、红色或蓝色。');
    if (start===end) return text;
    const code = ranges(text,codePattern), tokens=[];
    const tag=/<span\s+style=["']\s*color\s*:\s*#(?:222222|D62828|2563EB)\s*;?\s*["']\s*>|<\/span\s*>/gi;
    let plain='', colors=[], stack=[], cursor=0;
    for (const m of text.matchAll(tag)) {
      if (inside(m.index,code)) continue;
      const opening=m[0].match(open);
      if (!opening && !stack.length) continue;
      const chunk=text.slice(cursor,m.index); plain+=chunk;
      colors=colors.concat(Array(chunk.length).fill(stack.length?stack[stack.length-1]:null));
      if(opening) stack.push(opening[1].toUpperCase()); else stack.pop();
      tokens.push([m.index,m.index+m[0].length]); cursor=m.index+m[0].length;
    }
    const chunk=text.slice(cursor);plain+=chunk;
    colors=colors.concat(Array(chunk.length).fill(stack.length?stack[stack.length-1]:null));
    const offset=p=>p-tokens.reduce((n,[a,b])=>n+Math.max(0,Math.min(p,b)-a),0);
    let first=offset(start),last=offset(end);
    const math=ranges(plain,mathPattern), protectedCode=ranges(plain,codePattern);
    for(const [a,b] of math) if(first<b && last>a){first=Math.min(first,a);last=Math.max(last,b);}
    for(let i=first;i<last;i++) if(!inside(i,protectedCode)) colors[i]=color==='222222'?null:color;
    for(const m of plain.matchAll(/^[ \t]*(?:(?:#{1,6}|>|[-+*]|\d+[.)])[ \t]+)?/gm)) if(!inside(m.index,math)) colors.fill(null,m.index,m.index+m[0].length);
    for(const m of plain.matchAll(/^[ \t]*\|?(?:[ \t]*:?-+:?[ \t]*\|)+[ \t]*:?-*:?[ \t]*\|?[ \t]*$/gm)) colors.fill(null,m.index,m.index+m[0].length);
    for(let i=0;i<plain.length;i++) if(('\r\n'.includes(plain[i])||(plain[i]==='|'&&(i===0||plain[i-1]!=='\\')))&&!inside(i,math)) colors[i]=null;
    let output='';cursor=0;
    while(cursor<plain.length){let stop=cursor+1;while(stop<plain.length&&colors[stop]===colors[cursor])stop++;
      const value=plain.slice(cursor,stop),c=colors[cursor];output+=c&&c!=='222222'?'<span style="color:#'+c+'">'+value+'</span>':value;cursor=stop;}
    return output;
  }
  root.NectivonColors={applyColor};
  if(typeof module!=='undefined') module.exports=root.NectivonColors;
})(typeof window==='undefined'?globalThis:window);
