(function () {
  "use strict";
  const body=document.getElementById('body'),status=document.getElementById('status');
  let initialized=false, max=200000, serial=0, ack=0, renderSerial=0;
  let lastSent=null, externalRevision=0, handledFlush=null;
  function message(type,data){window.parent.postMessage({isStreamlitMessage:true,type,...data},'*');}
  function height(){message('streamlit:setFrameHeight',{height:document.body.scrollHeight+4});}
  function publish(flushToken=null,force=false){
    if(!force && body.value===lastSent)return;
    lastSent=body.value;
    message('streamlit:setComponentValue',{value:{text:body.value,serial:++serial,
      external_revision:externalRevision,flush_token:flushToken},dataType:'json'});
  }
  window.addEventListener('message',event=>{
    if(event.source!==window.parent||event.data.type!=='streamlit:render')return;
    const args=event.data.args, incomingRender=args.render_serial||0, incomingAck=args.ack_serial||0;
    if(initialized && (incomingRender<renderSerial || incomingAck<ack))return;
    renderSerial=incomingRender;
    // Colours repaint this frame without changing input, selection or revisions.
    if(args.palette && document.documentElement){
      const root=document.documentElement;
      for(const name of ['surface','text','muted','border','accent','soft','note_red','note_blue']){
        const color=args.palette[name];
        if(typeof color==='string' && /^#[0-9a-fA-F]{6}$/.test(color)){
          root.style.setProperty('--note-'+name,color);
        }
      }
      root.style.colorScheme=args.palette.scheme==='dark'?'dark':'light';
    }
    document.getElementById('label').textContent=args.label;
    body.setAttribute('aria-label',args.label);max=args.max_chars;body.maxLength=max;
    if(!initialized){
      body.value=args.value;lastSent=args.value;serial=incomingAck;ack=incomingAck;
      externalRevision=args.external_revision||0;initialized=true;
      publish(null,true);
    }else{
      ack=Math.max(ack,incomingAck);
      const explicitSourceChange=(args.external_revision||0)>externalRevision;
      if(explicitSourceChange){
        externalRevision=args.external_revision;
        if(body.value!==args.value)body.value=args.value;
        lastSent=args.value;
        handledFlush=null; // A prior flush with the old source revision will be rejected.
      }else if(ack>=serial && body.value===lastSent){
        // Unacknowledged/unsent input always wins over older page-render props.
        if(body.value!==args.value)body.value=args.value;
        lastSent=args.value;
      }
    }
    body.style.height=Math.max(160,args.height)+'px';height();
    if(args.flush_token!=null && args.flush_token!==handledFlush){
      handledFlush=args.flush_token;publish(args.flush_token,true);
    }
  });
  body.addEventListener('blur',()=>publish());
  body.addEventListener('input',()=>publish());
  for(const button of document.querySelectorAll('[data-color]')){
    button.addEventListener('mousedown',e=>e.preventDefault());
    button.addEventListener('click',()=>{
      const start=body.selectionStart,end=body.selectionEnd;
      if(start===end){status.textContent='请先选中要标注的文字或公式。';body.focus();return;}
      const updated=NectivonColors.applyColor(body.value,start,end,button.dataset.color);
      if(updated.length>max){status.textContent='标注后超过正文上限，未改动当前输入。';return;}
      body.setRangeText(updated,0,body.value.length,'preserve');
      const cursor=Math.min(start,updated.length);body.setSelectionRange(cursor,cursor);
      body.focus();status.textContent='选区颜色已更新，可预览或保存。';publish();height();
    });
  }
  new ResizeObserver(height).observe(body);
  message('streamlit:componentReady',{apiVersion:1});height();
})();
