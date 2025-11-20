import{d as g}from"./initial-query-slice-gsFlMgcr.js";import{A as $}from"./agent-state-u5yf9HVO.js";const u=g({name:"agent",initialState:{curAgentState:$.LOADING},reducers:{setCurrentAgentState:(t,e)=>{t.curAgentState=e.payload}}}),{setCurrentAgentState:D}=u.actions,U=u.reducer,w={url:"https://github.com/All-Hands-AI/OpenHands",screenshotSrc:""},m=g({name:"browser",initialState:w,reducers:{setUrl:(t,e)=>{t.url=e.payload},setScreenshotSrc:(t,e)=>{t.screenshotSrc=e.payload}}}),{setUrl:E,setScreenshotSrc:N}=m.actions,k=m.reducer;var d=(t=>(t[t.UNKNOWN=-1]="UNKNOWN",t[t.LOW=0]="LOW",t[t.MEDIUM=1]="MEDIUM",t[t.HIGH=2]="HIGH",t))(d||{});const I=[],y=g({name:"securityAnalyzer",initialState:{logs:I},reducers:{appendSecurityAnalyzerInput:(t,e)=>{const a={id:e.payload.id,content:e.payload.args.command||e.payload.args.code||e.payload.args.content||e.payload.message,security_risk:e.payload.args.security_risk,confirmation_state:e.payload.args.confirmation_state,confirmed_changed:!1},r=t.logs.find(o=>o.id===a.id||o.confirmation_state==="awaiting_confirmation"&&o.content===a.content);r?r.confirmation_state!==a.confirmation_state&&(r.confirmation_state=a.confirmation_state,r.confirmed_changed=!0):t.logs.push(a)}}}),{appendSecurityAnalyzerInput:C}=y.actions,H=y.reducer,c=1e3,p=["run","run_ipython","write","read","browse","browse_interactive","edit","recall","think","system"];function x(t){switch(t){case d.LOW:return"Low Risk";case d.MEDIUM:return"Medium Risk";case d.HIGH:return"High Risk";case d.UNKNOWN:default:return"Unknown Risk"}}const O={messages:[],systemMessage:null},f=g({name:"chat",initialState:O,reducers:{addUserMessage(t,e){const a={type:"thought",sender:"user",content:e.payload.content,imageUrls:e.payload.imageUrls,timestamp:e.payload.timestamp||new Date().toISOString(),pending:!!e.payload.pending};let r=t.messages.length;for(;r;)r-=1,t.messages[r].pending&&t.messages.splice(r,1);t.messages.push(a)},addAssistantMessage(t,e){const a={type:"thought",sender:"assistant",content:e.payload,imageUrls:[],timestamp:new Date().toISOString(),pending:!1};t.messages.push(a)},addAssistantAction(t,e){const a=e.payload.action;if(!p.includes(a))return;const r=`ACTION_MESSAGE$${a.toUpperCase()}`;let o="";if(a==="system"){t.systemMessage={content:e.payload.args.content,tools:e.payload.args.tools,openhands_version:e.payload.args.openhands_version,agent_class:e.payload.args.agent_class};return}if(a==="run")o=`Command:
\`${e.payload.args.command}\``;else if(a==="run_ipython")o=`\`\`\`
${e.payload.args.code}
\`\`\``;else if(a==="write"){let{content:s}=e.payload.args;s.length>c&&(s=`${s.slice(0,c)}...`),o=`${e.payload.args.path}
${s}`}else if(a==="browse")o=`Browsing ${e.payload.args.url}`;else if(a==="browse_interactive")o=`**Action:**

\`\`\`python
${e.payload.args.browser_actions}
\`\`\``;else if(a==="recall")return;a==="run"||a==="run_ipython"?e.payload.args.confirmation_state==="awaiting_confirmation"&&(o+=`

${x(e.payload.args.security_risk)}`):a==="think"&&(o=e.payload.args.thought);const n={type:"action",sender:"assistant",translationID:r,eventID:e.payload.id,content:o,imageUrls:[],timestamp:new Date().toISOString(),action:e};t.messages.push(n)},addAssistantObservation(t,e){const a=e.payload.observation;if(!p.includes(a))return;if(a==="recall"){const s=e.payload;let i="";if(s.extras.recall_type==="workspace_context"){if(s.extras.repo_name&&(i+=`

**Repository:** ${s.extras.repo_name}`),s.extras.repo_directory&&(i+=`

**Directory:** ${s.extras.repo_directory}`),s.extras.date&&(i+=`

**Date:** ${s.extras.date}`),s.extras.runtime_hosts&&Object.keys(s.extras.runtime_hosts).length>0){i+=`

**Available Hosts**`;for(const[l,S]of Object.entries(s.extras.runtime_hosts))i+=`

- ${l} (port ${S})`}s.extras.repo_instructions&&(i+=`

**Repository Instructions:**

${s.extras.repo_instructions}`),s.extras.additional_agent_instructions&&(i+=`

**Additional Instructions:**

${s.extras.additional_agent_instructions}`)}const h=`OBSERVATION_MESSAGE$${a.toUpperCase()}`;if(s.extras.microagent_knowledge&&s.extras.microagent_knowledge.length>0){i+=`

**Triggered Microagent Knowledge:**`;for(const l of s.extras.microagent_knowledge)i+=`

- **${l.name}** (triggered by keyword: ${l.trigger})

\`\`\`
${l.content}
\`\`\``}const _={type:"action",sender:"assistant",translationID:h,eventID:e.payload.id,content:i,imageUrls:[],timestamp:new Date().toISOString(),success:!0};t.messages.push(_);return}const r=`OBSERVATION_MESSAGE$${a.toUpperCase()}`,o=e.payload.cause,n=t.messages.find(s=>s.eventID===o);if(n){if(n.translationID=r,n.observation=e,a==="run"){const s=e.payload;s.extras.metadata.exit_code===-1?n.success=void 0:n.success=s.extras.metadata.exit_code===0}else if(a==="run_ipython"){const s=e.payload;n.success=!s.content.toLowerCase().includes("error:")}else(a==="read"||a==="edit")&&(e.payload.extras.impl_source==="oh_aci"?n.success=e.payload.content.length>0&&!e.payload.content.startsWith(`ERROR:
`):n.success=e.payload.content.length>0&&!e.payload.content.toLowerCase().includes("error:"));if(a==="run"||a==="run_ipython"){let{content:s}=e.payload;s.length>c&&(s=`${s.slice(0,c)}...`),s=`${n.content}

Output:
\`\`\`
${s.trim()||"[Command finished execution with no output]"}
\`\`\``,n.content=s}else if(a==="read")n.content=`\`\`\`
${e.payload.content}
\`\`\``;else if(a==="edit")n.success?n.content=`\`\`\`diff
${e.payload.extras.diff}
\`\`\``:n.content=e.payload.content;else if(a==="browse"){let s=`**URL:** ${e.payload.extras.url}
`;e.payload.extras.error&&(s+=`

**Error:**
${e.payload.extras.error}
`),s+=`

**Output:**
${e.payload.content}`,s.length>c&&(s=`${s.slice(0,c)}...(truncated)`),n.content=s}}},addErrorMessage(t,e){const{id:a,message:r}=e.payload;t.messages.push({translationID:a,content:r,type:"error",sender:"assistant",timestamp:new Date().toISOString()})},clearMessages(t){t.messages=[],t.systemMessage=null}}}),{addUserMessage:L,addAssistantMessage:R,addAssistantAction:T,addAssistantObservation:G,addErrorMessage:W,clearMessages:z}=f.actions,b=t=>t.chat.systemMessage,K=f.reducer;export{d as A,L as a,W as b,z as c,E as d,N as e,b as f,R as g,G as h,w as i,C as j,T as k,H as l,U as m,K as n,k as o,D as s};
