// Optional developer check: Node + ws, Chrome with --remote-debugging-port=19222.
// Not required for students or the runtime GUI.
const http = require('http');
const fs = require('fs');
const WebSocket = require('ws');
const expression = process.argv[2] || '({title:document.title,status:document.getElementById("status").textContent,samples:document.getElementById("sample").options.length,images:[...document.images].map(i=>({id:i.id,width:i.naturalWidth}))})';
http.get('http://127.0.0.1:'+(process.env.DEBUG_PORT||'19222')+'/json', r => {
  let data='';r.on('data',b=>data+=b);r.on('end',()=>{
    const target=JSON.parse(data).find(t=>t.type==='page' && (process.env.PAGE_URL ? t.url.includes(process.env.PAGE_URL) : process.env.DEBUG_PORT ? true : t.url.includes('18765')));
    if(!target)throw Error('GUI tab not found');
    const ws=new WebSocket(target.webSocketDebuggerUrl);let id=0;const pending=new Map();
    const send=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}));});
    ws.on('message', b=>{const m=JSON.parse(b);if(pending.has(m.id)){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result);}});
    ws.on('open',async()=>{try{
      await send('Emulation.setDeviceMetricsOverride',{width:1360,height:1100,deviceScaleFactor:1,mobile:false});
      const result=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
      console.log(JSON.stringify(result,null,2));
      if(process.argv[3]){const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});fs.writeFileSync(process.argv[3],Buffer.from(shot.data,'base64'));}
      ws.close();
    }catch(e){console.error(e);ws.close();process.exitCode=1;}});
  });
});
