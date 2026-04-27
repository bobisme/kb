---
type: source
title: 2026-04-07 LiveRamp USB Team Intro Transcript
recording_date: 2026-04-07
audio_file: GMT20260407-143343_Recording.m4a
duration_seconds: 4493.1
speakers_detected: 6
asr_model: large-v3
diarization_model: pyannote/speaker-diarization-community-1
language: en
---

## Speakers

- @speaker_03: unknown
- @speaker_unknown: unknown
- @speaker_04: unknown
- @speaker_02: unknown
- @speaker_00: unknown
- @speaker_01: unknown

# Transcript

## @speaker_03 [00:00:01 → 00:00:53]

proposal doc. But Bob's going to be coming on to help us kind of build out some POCs on this local stack digital twin type service, mock service type system. So this meeting is really for Bob to understand. So I'm going to let Bob drive questions and everything. And Bob, please ask Xiaodong anything you need to know or what kind of direction you want so you can understand the system and like what level, etc. And then Xiaodong can get into talking through any of those and answer any questions. And Xiaodong is the team lead for the segment building team. And so that would be that their team interacts with many teams.

## @speaker_unknown [00:00:57 → 00:01:01]

will be a very beneficial thing for them. Cool.

## @speaker_04 [00:01:01 → 00:01:16]

Want me to? No, not me. So did the rest of you get a chance to kind of discuss what the initiatives are? Or should I talk about those? Or are there any questions about that before I dig into the list of questions I have?

## @speaker_02 [00:01:20 → 00:01:25]

Yeah, Bob, we didn't discuss too much. It's good for you to share context first.

## @speaker_00 [00:01:26 → 00:01:26]

Yeah.

## @speaker_02 [00:01:27 → 00:01:28]

I don't want to go too detail.

## @speaker_04 [00:01:29 → 00:01:42]

Oh, okay. Sorry. Keep coughing and stuff. So the context is, well, it sounds like you've shared the proposal that I sent Sean. Is that correct?

## @speaker_03 [00:01:44 → 00:01:53]

Yeah. I shared a summary of it in a Slack message and doc itself, but just a summary of the week one, week two, week three kind of, and like ramp out of box type thing.

## @speaker_04 [00:01:55 → 00:02:42]

Yeah. Okay. So that's the goal is like, I'm coming on to do a proof of concept to try to help teams. My pitch to Sean was more about like broader, you know, agentic development and autonomous development infrastructure. It's important to, before we go down that path, get some guardrails in place so that we can enable that. And the beginning part is to like, make sure that we can do this, I guess, live rip in a box is the, you know, big goal, if that's possible and getting along that way is making sure it's like the testing infrastructure is good for, I mean, really long-term goals agents, but it needs to be good for like software engineers first. Does that sound accurate?

## @speaker_00 [00:02:47 → 00:02:48]

Yeah. Yeah. Yeah.

## @speaker_01 [00:02:52 → 00:02:53]

Okay.

## @speaker_00 [00:02:55 → 00:03:00]

Let me quickly share my screen. Oh, sure. Just one second.

## @speaker_01 [00:03:02 → 00:03:05]

Yeah, here we go.

## @speaker_00 [00:03:05 → 00:03:29]

Yeah, can you guys see my screen? Yes. Let me share the screen, the mural in the Zoom chat. Not sure. Yeah, just let me see in the Zoom chat.

## @speaker_01 [00:03:31 → 00:07:03]

Yeah, you can try to see whether you can open it or not. Yeah, let me get started. So basically, And the service, our product is USB, the LightRamp new segment building, that product. And here is kind of like a high level architecture for the segment building product. From as a top level, we have the application layer, right? In the application layers, we have the clean room, USB. USB is our segment building, the new segment building. And the chatbot, today we have AI-powered segment building, which is built on top of the USB. And we have a centralized metadata service, which is a catalog. And we have a third-party data marketplace application. And as the second layer is the service layer, and which we have corresponding backend service support the front end use case. And so in the control plan, we have the the different control plans for USB building use cases. And for clean ROM use case or Snowflake use case, we have Ferris Circle, Segment API, Segment Engine, and Segment Injection API to support the segment building functionalities. And however, for the connect use case, we have the data circle, segment API, DSM, action runtime, action circle as a backend service to support the USB features. So on the data layer, the warehouse layer, we have the BigQuery, Snowflake, and the single store. So how do we interact between each components, right? We have the event bus, the pop-stop, the segment pop-stop. When a segment is created, the downstream system will receive the message from the pop-stop. And when a new author, a table author get updated, then we have corresponding magnesium to refresh the segment accordingly. And also we have other, the DL marketplace events, right? When a buyer purchase a segment on DL marketplace, then there is a pops up amazing is sent by DL marketplace as well. So we also have the temporal workflow to support the orchestrate or the asynchronous job performer on top of the single store. Okay, that's pretty much about a quick introduction about our single building high-level architecture. Yeah, any questions, please feel free to ask.

## @speaker_04 [00:07:03 → 00:07:18]

Yeah. So what is it? I'm trying to remember from back in my LiveRamp days, what exactly is the segment builder as a product? Is that when somebody logs into a UI and then they choose?

## @speaker_01 [00:07:18 → 00:07:25]

Yeah, yeah. I think that the segment building is kind of like, let me quickly share with you

## @speaker_00 [00:07:28 → 00:07:46]

Oh, connect looks different.

## @speaker_01 [00:07:46 → 00:08:00]

Yeah, this is, yeah, our LiveRamp connector platform. Within its platform, there is a beauty segment button. There is an entry. You can just click it.

## @speaker_00 [00:08:03 → 00:08:18]

Yeah, here we go.

## @speaker_01 [00:08:19 → 00:08:22]

Yeah, this is our segment building product.

## @speaker_00 [00:08:25 → 00:08:28]

It will take several seconds to finish the loading.

## @speaker_04 [00:08:30 → 00:08:40]

So I remember years ago, they were talking about replacing Connect, the Ruby on Rails application, with something else. Has that happened?

## @speaker_01 [00:08:42 → 00:08:50]

Yes, it's happening recently. And let me finish this one first.

## @speaker_03 [00:08:50 → 00:09:21]

OK. Yeah. It's in the progress. Um, honestly we gave up on like completely replacing stuff. And so we're new things are built on top of, um, the react front end and stuff and uses, um, other stuff, but there's still Ruby on rails and still, uh, Ruby like RLDB and stuff like that, that are around. Um, but there's areas of modernization that have happened.

## @speaker_01 [00:09:23 → 00:11:14]

Yeah, exactly. So here, this is our new segment of building. From here, you can check all the available data sets you can use to build the segment. And this is the cleanroom data. CR means cleanroom. We are supporting cleanroom data, the cleanrooms table, and then fill in the, choose the value to give a name. You can also perform the size calculation, then save it. After that, a segment, the segment will be displayed on the catalog segment, the segment list page. Yeah. From here, you can see the segment you just created. And yeah. So at a product level, here, this is the segment list page. You can find the segment you just created under the build segment folder. And regarding the legacy segment builder, this is a legacy segment builder. We call it GSP. So far, we already migrated several customers, the DSP customers to USB. And recently, we are working on the implementation for migrating another legacy segment builder, CP2, to USB. But that hasn't happened yet. We are still working on it. Yep.

## @speaker_04 [00:11:15 → 00:11:22]

OK, so is this the whole product level of the USB?

## @speaker_01 [00:11:23 → 00:13:14]

Oh, yeah. No, no. This is just a quick introduction. From here. Yeah, yeah, I think the you can also you can also choose right, you can also choose the segment you you created before and to conduct another future combination by it is a segment. Some whole some Holden Holden created a today, we can also use it to build another new segment. And also, within the segment building, we also support the Data Marketplace data. The Data Marketplace is another individual product, which a seller party segment, right? The seller can sell his segment to the marketplace, then buyer can purchase the segment he want from the marketplace. Then after the buyer purchase that segment, the segment will be available under this folder and the user can choose that segment to see, oh, the price, right? And the uh the price and the and the some some uh you know the media share percentage right and also user can combine the third party segment with with the other uh first party segment first body segment means the the customer's own data yeah uh Yeah, any questions?

## @speaker_04 [00:13:15 → 00:13:27]

Well, it was more, one thing about the USB team is like this, the segment builder is like your whole product, like you don't have other products.

## @speaker_01 [00:13:28 → 00:14:54]

Yeah, we have another product, which is the AI-powered segment building. From here, you can type the segment building prompts like the purchase, like shampoo, blah, blah, blah. Okay. Yeah, it will navigate the system, the user will be navigated to another product, the AI chatbot, and to start his assembly building journey where the natural language, by the natural language input chatbot from here, you can type your segment building input there. At the back end, we have the AI, the long graph to orchestrate the whole segment building flows. Yeah. Yeah, this part is also owned by the USB team. Yeah. I'm not sure whether this org has any data backfilled in the virtual DB? Probably not. Any questions?

## @speaker_04 [00:14:56 → 00:15:18]

I mean, I have a list of questions I can go through. Yeah. I mean, the first was going through what USB team is the segment builder. I guess we kind of talked about that. I also requested access, if you're allowed to share it, to that graph, the architecture layout.

## @speaker_01 [00:15:19 → 00:15:42]

Yeah. Yeah, I think the ‑‑ I can share the ‑‑ if I understand the question you just asked is whether we can provide ‑‑ we can grant you some access to the system, right? Yeah, definitely. We can have you ‑‑ I was talking about the Miro graph specifically. Oh, the Miro. This one, right?

## @speaker_00 [00:15:42 → 00:15:44]

Oh, yeah. I think I already ‑‑

## @speaker_04 [00:15:46 → 00:15:50]

You sent me the link, but I don't have a live ramp email address, so I have to.

## @speaker_01 [00:15:50 → 00:15:54]

Oh, OK. Maybe I can.

## @speaker_04 [00:15:54 → 00:15:56]

Or you can send me screenshots or something in email.

## @speaker_01 [00:15:56 → 00:16:00]

Yeah, I can export it to maybe a PDF or.

## @speaker_02 [00:16:05 → 00:16:10]

You can leave that to me. I will share it with Bob. Yeah.

## @speaker_01 [00:16:11 → 00:18:30]

Yeah, sure. Thank you. Thank you. Thank you. Thank you, Josh. Yeah. OK, let me continue. Yeah. Yeah, I think we already finished the high-level introduction. Then let's move on. detail, not detail, right, the medium level, right? So in this diagram, in this mirror, you can see we have connected use case, and clean room use case, and the snowflake use case. Those three use cases are all the use cases we support in the USB so far. So, and as you can see here, different color means different use case, right? The green means the connector use case. You can see, oh, I have building, a user start building the segment on USB and The purple color means that all those are the common component, right, across different use cases, right? But as the control plan, right, they are different, right? The connected use case have all components, right, in this green component. under this green area, right? They are supporting the connect same rebuilding use case, right? The red area means supporting the clean room use case. And the yellow area, the components in the yellow area are supporting the snowflake use case. And the different use case, their data also store into different warehouses. clean room, the data, the connector, the data actually is already stored into the single store cluster. However, like the clean room, we are still storing data into the legacy store, the Swift store. And in the Snowflake, right, actually the data is, the data is still in their own warehouse.

## @speaker_04 [00:18:33 → 00:18:42]

OK. So I'm curious, like, what does the USB team own in this chart? Like, is it just?

## @speaker_01 [00:18:42 → 00:19:09]

Yeah. Yeah. I think this part is like this one, the USB API. And we also have some backend service and another so-called translation service. I didn't put it in this middle, but under those components, we have the...

## @speaker_04 [00:19:09 → 00:19:16]

Sorry, you said you own the USB API, and what was the other thing?

## @speaker_01 [00:19:17 → 00:19:28]

Yeah, the other thing is, let me jump into this area. We have the UI and the USB and the second engine.

## @speaker_00 [00:19:30 → 00:19:31]

Let me quickly.

## @speaker_01 [00:19:33 → 00:20:08]

Yeah, those components highlighted in blue color, right? And catalog. And this part. This part, partial OVH, right? This is, the file server is a very, very complex service and consists of multiple services. So one of the service already, you know, transfer, I mean, the transition to USB team.

## @speaker_00 [00:20:08 → 00:20:19]

And so I need to use maybe this color, but... Something like that.

## @speaker_01 [00:20:20 → 00:20:34]

Yeah. And yeah, that's pretty much right. There's another, which is the segment building the USB job. Job.

## @speaker_00 [00:20:36 → 00:20:39]

And let's queue. Okay.

> [pause: 6s]

## @speaker_01 [00:20:46 → 00:20:54]

Yeah. Those green, those components with green color are USB services.

## @speaker_00 [00:20:55 → 00:20:55]

Yeah.

## @speaker_01 [00:20:57 → 00:20:58]

Okay.

## @speaker_00 [00:20:58 → 00:20:58]

Yeah.

## @speaker_01 [00:20:59 → 00:24:55]

Yeah. Let us to jump into the, the more, you know, low, low level, right? The bottom level, right? The detail level. So what, what functions does USB have today? In USB, we have a few suggestions. what does field execution mean? So basically, when user start building the segment in USB, when he choose a field, like the age, so basically we have the function to automatically to get the value, the all distinct values of that edge field, right? Then render on the, I mean, the display on the USB UI, let user to automatically, let a user can easily to choose, oh, which field, which field value user can choose. want to use. Let me quickly share with you. Oh, sorry. Yeah, let me quickly share with you what does the field suggestion mean. So basically, for example, like if I choose this one, so this is the suggestion. So when I choose the banner, user, if we don't have this kind of suggestion, the way it was right, user never know, oh, what way do I should fill in for banner is filled, right? So we have the mechanism to automatically to calculate all these things the way it was. a banner and display on this drop-down list UI. Let them easily to choose, oh, I want to choose this value and another value. It supports the multiple choices. Yeah. Yeah, this kind of, this is, we call this field suggestion function. So as I just mentioned, USB supports clean room, snowflake in the single store. So clean room and the single store, they share the same field suggestion approach. Let me jump into this view. Yeah, this diagram. So start from the UI. then once the user click a field right, the UI will send a request to the backend. Then the backend will cause a fat circle to run a query like the select distinct banner from table A. Right? So this query will be executed on the, if the customer is Snowflake, their data is hosted in the Snowflake warehouse, then the query will be executed on the Snowflake warehouse. and we have some cache mechanism, right? So if user, previously user already clicked that field, then at the backend we store, we will store the values into the cache. Then next time when user choose the same field again, right, then we just pull the data from the cache and quickly she owns the UI. Yeah. Yeah.

## @speaker_04 [00:24:56 → 00:24:58]

Is the cache on the USB backend?

## @speaker_01 [00:24:59 → 00:26:49]

Yeah, in the cache, in the USB backend. Yeah. So in the single store, I think the single store is pretty simple. Yeah, single store. So in the single store, single store means the customer, provided the data to LightRamp, and the data actually, the customer data actually is hosted in the LightRamp hosted warehouse, which is a single store. So the data is, the customer's data is going through the LightRamp injection pipeline, uh then eventually all data will be stored in the single store so we have the magnesium in the uh ingestion side right they will automatically to detect all the values of all fields in customer's file, like the banner I just mentioned. After the data is injected to LightRamp, then in catalog side, they will have all values for customer's file. of that field stored in catalog side, right? So we just, from UI, we just request to have a very simple API course to catalog to get all values for that field and present on the UI. We don't need to run another query to get the distinct values of that field like we are doing for cleanroom and Snowflake. Yeah. OK. Yeah, please, Bob, please feel free to interrupt me if you have any question.

## @speaker_04 [00:26:52 → 00:26:55]

That's OK. I don't think I have any questions, but yeah.

## @speaker_01 [00:26:56 → 00:31:29]

Yeah, I think the, I'm not sure whether you know catalog service, if you know catalog service before or not. Yeah, if no, I can also. Oh, OK, cool. Yeah, catalog is kind of like a unified or centralized metadata service. No matter the data is created in USB or the data is ingested by customer, you can understand customer have all kinds of information, no matter it's table, data set, segments, or even destinations. Yeah, that's the principle we are always stick to, right? Let all applications store the data into catalog so that we don't need to check with each operation team or whether, how do we get the data? We just needed to interact with catalog service. That's the biggest benefits that we get. Okay. Let me talk about segment creation. Segment creation is a very important use case. For snowflake and single store, when user create a segment, right? And the request will be sent to the USB side. Then in the USB side, we will create a segment in DSM side. Then create an audience in the DSM side. Here is the step. The second step is create an audience in DSM. Audience means actually the latest name we call is a data set, right? But in the legacy connector world, you know, most people call it audience, right? But in the new world, we call it data set. It's kind of like a container, right? You know, or a populator, or data table segment, right? So basically when user first, when user create a segment initially, then at the backend, we need to create that kind of container to populate the segment the user is creating. Then after the audience is created, then USB will call the segment API to create the segment in the IRTP, the legacy field ID, value ID. Yeah, so then there is a sync between SignalCore and the catalog. And because the simple core is kind of a legacy service, right? But we still can't get right out of it. But how do we make sure the new metadata service, the catalog service, can have all the information, they are building a synchronized network. flow between segmental core and catalog. Once a USB creates the segment in segmental core, the segment will be automatically synced to catalog as a segment assert. But we still will have another separate code to catalog, to patch the same building metadata, right? Because the same building metadata is a new, those are new information that we have in USB, right? But in the legacy segmented building, we don't have those informations. But how do we store those kind of new informations from USB? Then we have a separate course to patch the segmented building metadata into the segment R cert in catalog.

## @speaker_04 [00:31:34 → 00:31:39]

Are you saying that? Actually, it doesn't matter, the question.

## @speaker_01 [00:31:40 → 00:31:43]

Yeah, please feel free to ask questions.

## @speaker_04 [00:31:44 → 00:31:58]

Yeah. I'll let you keep going. I think the more important questions I have are after this, when we talk about protocols and testing and stuff.

## @speaker_01 [00:31:59 → 00:35:34]

OK. OK, let me move on. So yeah, so in the cleanroom and the cleanroom use case, the same equation. Let me jump to the cleanroom part. Yeah, cleanroom actually, cleanroom, it includes two different use cases, right? The first one is a 2P use case, right? Second one is a third-party use case, right? But in USB, at the backend, there are not too much difference between each other. So first step is USB receive the request from UI, then Same with the previous one, we also call segmented call to create the segment, then catalog the segmented call, sync the segmented metadata to catalog, then we have the patch to segmented call. But the difference is here, because if the the data is kind of clean room third-party data, then we needed to have better marketplace to register, not register, create a DMS segment for this segment user created in the USB in IODB. Because as long as the data, the clinical data is third party, we needed to invoke data marketplace and ensure there is a DMS segment created in the IODB so that when user activated that segment, we have flow to ensure the segment is the segment will be activated through third-party activation flow. Okay. Yeah, let me move on. Yeah, the last one about the segment creation is a single-store third-party. Let me jump to the middle. Yeah, so this is... third-party segment. So we receive the request from UI, then USB create a container, the data set in DSM, then create a segment in segment call, and the segment call legacy field ID in ILTP, then USB There is a synchronization between signal core and the catalog, the USB to patch the segment building metadata in catalog. Then we have the data marketplace involved to register the third-party metadata into the second asset in catalog. Okay. Yeah. Bob, any questions regarding the segment operation?

## @speaker_04 [00:35:35 → 00:36:00]

Not yet. I'm more thinking about... So when I'm looking at this, I'm looking at all the different services that your backend services are connecting to and thinking about the testing boundaries. So if you talk about the segment materialization, do you have other services that you're talking to? Or is it the same kind of services in different ways?

## @speaker_01 [00:36:01 → 00:37:23]

Yeah, I think the segment creation is kind of very straightforward, right? Kind of like... inter-system interaction at the metadata level, right? Because we send the creation chart to ensure, oh, there is a segment author created in catalog, and the user can see that segment in catalog, right, in the UI. But on the backend, where do we store the segment data? And how do we generate the segment data? the segment data based on the logic user, based on the rule user created for his segment in USB, that's belong to the segment materialization, which is more complicated than segment creation. So let me quickly jump into the segment materialization So the segment of materialization regarding the snowflake part, You know, after the segment is created, right, by USB, then... Xiaodong, sorry to interrupt.

## @speaker_02 [00:37:23 → 00:37:39]

So Xiaodong, I need to jump to different meeting. So I will transfer the host to you, okay? And for the material, we need forward to Bob. So you can send me the link of this meeting. I will figure a way to how to share material with Bob, okay?

## @speaker_01 [00:37:39 → 00:37:41]

Okay, okay. Sure, thank you.

## @speaker_02 [00:37:41 → 00:37:42]

Thank you, Bob.

## @speaker_01 [00:37:43 → 00:39:36]

Thank you, Josh. Yeah, let me continue. So basically, when a USB segment is created, right, when a segment is created by USB, then USB will call file circle API to start the segment materialization, then Remember, the segment actually is represented by a circle statement. So you can understand the materialization is kind of like executing a circle statement and getting the result stored somewhere. Yeah, so for Snowflake, the FedEx Circle will play a circle execution engine role, right, to execute the segment query. Here, the second step is FedEx Circle will translate the circle and run it on the LightRamp hosted Snowflake account, then return the execution result in a file, a data file. Then we have the segment engine on by USB to ingest the data into segment API. Then segment API store the data into the legacy segment store. So then notify USB, oh, the segment is materialized. The user can see the segment has changed from updating to ready on the UI. Yeah.

## @speaker_unknown [00:39:36 → 00:39:36]

OK.

## @speaker_01 [00:39:37 → 00:42:02]

So basically you can understand, uh, uh, if we are talking about, uh, component level for Snowflake, in this case, the segment materialization involves by, uh, the USB signal engine, ferrous circle, then the core and the, and the catalog. Yeah. Okay. Yeah. And let me jump into the, the. materialization, the single-store part. So in the single-store part, some components are same with Snowflake, but there are some different components involved. So when a segment is created by USB, then uh, the segment, uh, is registered, registered in catalog aside, then DSM, the transit management team, their service will, uh, will be informed, right? By, by, by, by pop star. They, they, they will call the, call the, they will call the file circle to, they will call the file circle to translate the segment circle statement. Then send the auction API to auction circle. The auction circle is in charge of materializing the segment in single store. Then the DSM will periodically call the API to check whether the segment is still being materialized or the materialization is completed. Then update it to segment call or the segment status needs to be changed from updating to ready. Ready means the segment is materialized successfully. Yeah, basically, you can understand, oh, for snowflake, for single-throw materialization, the new world, same with the core, catalog, DSM, action runtime, action circle, and the file circle. Yeah.

## @speaker_04 [00:42:03 → 00:42:09]

Okay. So I don't see any lines between USB and that. Do you communicate through PubSub, or is it?

## @speaker_01 [00:42:10 → 00:43:04]

Yeah, yeah, we we just need a USB. Actually, yeah, I think the USB just register the segment in catalog, then or downstream or the or the subsequent steps, right are triggered by the, the catalog where the puffs up right when a segment of quick in catalog side, they have magnesium to monitor the segment creation event, right? Then send the pop-up to DSM. Oh, there is a segment created just now. Then DSM will play an orchestration role, orchestration engine role to cause a file circle to make the circle translation, then some be an API to action circle for segment materialization.

## @speaker_04 [00:43:06 → 00:43:06]

Okay.

## @speaker_01 [00:43:08 → 00:43:12]

And then you had one more. Oh, sorry.

## @speaker_04 [00:43:12 → 00:43:13]

You had one more use case.

## @speaker_01 [00:43:14 → 00:44:28]

Yeah. Actually, the last use case is regarding the segment materialization for clean room use case. Actually, it's very similar with Snowflake use case. The only difference is the warehouse. Uh, yeah, the USB, uh, a segment that is created by USB, then USB will co-file a SQL to, uh, execute the, for executing the query on LightRambler hosted BigQuery, uh, and, uh, export the result to a GCS, uh, data file. Then the same engine, right? Same engine copy the file. then copy the file results, require query result file, and then ingest into second API. Then second API publish a message to USB when the ingestion completes. Then the USB, after the ingestion is completed, USB will update the second instance from updating to ready. Yeah.

## @speaker_00 [00:44:28 → 00:44:29]

That's a lot.

## @speaker_01 [00:44:29 → 00:45:34]

That's a whole lot. Yeah. So that's pretty much. Yeah. So as you can see here, USB. So this is I understand Sean's position, right? USB is different with other product, right? we have too much, we have a lot of dependencies, right, on the, on other teams, right, other services, right, because at the high level, we have three different use cases, and at the backend, right, different use cases require different dependency services. Yeah, yeah, but we are, we are planning some unification works in the recent delay, but still, but you know, as the near term, we still have to face this kind of situation or a bunch of dependency dependencies, dependencies to support different use cases. Yeah.

## @speaker_04 [00:45:36 → 00:46:00]

Yeah. We've got about, oh my gosh, we only have about 11 minutes. I mean, I can go late if you can, but let me try to get through some of the questions I had. So what kind of, oh, sorry. Go ahead. Like I said, what kind of protocols? I see Thrift on there. Looks like you're doing some raw SQL. I'm guessing, right? Yeah. Do you have like REST, gRPC, any other stuff?

## @speaker_01 [00:46:00 → 00:46:36]

Yeah. Yeah, I think the REST API and the and the PubSub, yeah, I think from USB side, right, we only have those two protocols. But on the backend, like the RPC or other protocols, they have, but in USB, we only have REST APIs and the PubSub. Those are the only... USB only has...

## @speaker_04 [00:46:36 → 00:46:41]

Wait, you mean you serve? Or wait, never mind. I see. Yeah, okay.

## @speaker_01 [00:46:41 → 00:47:29]

Yeah, yeah, yeah. Like all use cases, we just... Oh, we submit an API request to save the core catalog. And sometimes we need to subscribe a specific, a particular pop-up topic to receive the message to refresh the... to re-materialize the segment when the upstream table is updated, the soft table is updated. Those are only two protocols we have in USB, but in the dependency services, I remember they have other protocols except PubSub and REST API.

## @speaker_04 [00:47:30 → 00:47:39]

So all the services that you call to, they only use REST and PubSub? Yeah, exactly. Even the FedSQL?

## @speaker_01 [00:47:40 → 00:47:49]

Yeah, even the FedSQL. We are also calling the API to materialize segment or translate the segment, the so-called statement.

## @speaker_04 [00:47:50 → 00:47:56]

Okay. How big is the code base for USB, roughly? Do you know?

## @speaker_01 [00:47:58 → 00:48:07]

If you just basically, do you want to include the front end or just the back end?

## @speaker_04 [00:48:09 → 00:48:12]

I mean, if you know separately, that's useful.

## @speaker_01 [00:48:13 → 00:48:58]

Oh, OK, yeah. In the back end, we have the USB back end, USB API, USB job, and the matricule. And, and also like to save the engine. And we recently we just took over the Ferris circle translation API. Yeah, I think the world I think the compare with I think the probably, it's very difficult to say, right? Yeah, that's fine.

## @speaker_00 [00:48:58 → 00:48:59]

Yeah, yeah.

## @speaker_01 [00:48:59 → 00:49:56]

So basically, maybe medium level, not really, because we just own... two projects, right? The one is USB, another is MSB. MSB is a new product that we just created last year, but USB already has more than four years, right? We created that project for more than four years. But given it's just a single product, right? The USB is just one single product, so the repo is probably just a medium level. How can I say that? The cool line or the volume? Yeah, just a very medium level, not a large level. Medium level.

## @speaker_04 [00:49:56 → 00:50:00]

Do you have separate repositories for each service, or is it all one?

## @speaker_01 [00:50:01 → 00:51:40]

Yeah, I think the... Yeah, this is the MSP project. Let me share with you the USB repository. Yeah, we are the monorepo. This is our repository, the backend repository, the same building backend. From here, you can see, oh, there is a same building API, building job, right? The API is kind of like a controller or so, right? a job handles with all kinds of asynchronous jobs. And the matricule is a standard alone service, which is responsible for translating the UI building logic to StarCross element. And yeah, and there is segment engine, segment engine service. Yeah, the segment engine service, this one, the final segment engine service, which plays a role to execute, to execute the SQL, circle on Snowflake warehouse or BigQuery warehouse.

## @speaker_00 [00:51:42 → 00:51:42]

Yeah.

## @speaker_unknown [00:51:43 → 00:51:43]

Okay.

## @speaker_04 [00:51:44 → 00:51:57]

Yeah. Sorry. I'll get to the next one. What does your local development setup look like? Like can you run the services on your laptop now or do you have to launch it?

## @speaker_01 [00:51:59 → 00:52:53]

Uh, yeah, I think the weekend we are able to run our service locally is kind of a job because our service, the, the job, right. The job is kind of, uh, uh, you know, spring boat service. You can just, uh, just run this application. Yeah. We can, we can run it on our locally, but probably, uh, I, I, I don't have the, I didn't connect our company's VPN, so the service can connect to our database, but after you connect it to the VPN, and then you can run the service, right, the Spring Boot application locally. Yeah.

## @speaker_04 [00:52:53 → 00:52:57]

Okay. Is everything Java?

## @speaker_01 [00:52:59 → 00:53:11]

Uh, yeah, I think the most of the service on Java, uh, except the Metricure, uh, they are using the, they are using the, they are using the, uh, Kotlin.

## @speaker_00 [00:53:13 → 00:53:15]

Yeah. They are using Kotlin. Yeah.

## @speaker_01 [00:53:15 → 00:53:27]

And the same engine they are using Go, as you can see here, Go. Yeah. Yeah, and we have the Fire Circle Translation API.

## @speaker_00 [00:53:27 → 00:53:29]

They are using Scala.

## @speaker_01 [00:53:31 → 00:53:35]

Yeah, but you can, the major language in USP are Java.

## @speaker_04 [00:53:37 → 00:53:40]

Which one did you say was Scala? You said the translation.

## @speaker_01 [00:53:40 → 00:53:44]

The Fire Circle, the translation. Let me share with you. Scala.

## @speaker_00 [00:53:44 → 00:54:10]

Translation. Fire Circle Translation. Uh, is every name the repository? Oh, I think it's in the EF. Oh, this one. There is a service.

> [pause: 11s]

## @speaker_01 [00:54:21 → 00:54:41]

I need to check with them which branch they are using, which branch is the latest one. That's OK. Yeah, yeah, yeah, yeah. I think I can get back to you. Because this service is just what we took over from them last quarter. So yeah.

## @speaker_04 [00:54:42 → 00:54:53]

Okay. So can you tell me about like, what testing looks like for these services? Like, what does your test suite look like? You have unit tests, integration tests?

## @speaker_01 [00:54:54 → 00:55:55]

Yeah, we have, firstly, we have the UT, right? We have the UT. You know, whenever we want to perform the deployment, we will or the UT will be executed automatically during the CIC, in the CIC pipeline. And then we also have the automation test cases, right, running regularly on the production environment. But that part is on, this kind of thing is on by QET. Yeah, and they will automatically, they will notify us whenever any test cases executed fail, we will be informed by them, yeah.

## @speaker_04 [00:55:55 → 00:55:56]

What's the other team you said?

## @speaker_01 [00:55:57 → 00:56:01]

The QE team, the Joshua's team. Oh, okay.

## @speaker_04 [00:56:01 → 00:56:02]

Oh, the QE, okay, gotcha.

## @speaker_01 [00:56:03 → 00:56:32]

Yeah, QE, quality engineer. They build a lot of very useful test cases and running on production periodically. You know, the frequency depends on the case priority, or whether it's a high priority, then running hourly, and if it's a medium priority, then maybe two times per day. Yeah, depends on the

## @speaker_04 [00:56:34 → 00:57:07]

priority the feature priority that's interesting but do you have any uh so what i've been thinking about is well i guess uh shannon talked about it with like live ramp in a box is there anything close to that where you can run like a kubernetes cluster on your laptop with your service connect to mock services, all the things it talks to and like actually run tests against those?

## @speaker_01 [00:57:09 → 00:57:10]

No, we don't have such.

## @speaker_04 [00:57:11 → 00:57:11]

Okay.

## @speaker_01 [00:57:11 → 00:57:24]

Yeah. Yeah. We don't have such, you know, blocks to containing all the dependent service as a mock, right? Yeah.

## @speaker_04 [00:57:24 → 00:57:25]

I mean, it's a hard thing.

## @speaker_01 [00:57:27 → 00:57:32]

Yeah, yeah, I think that's very useful.

## @speaker_00 [00:57:35 → 00:57:39]

If we have that, that should be very useful, very helpful.

## @speaker_04 [00:57:40 → 00:57:46]

I think so. I hope so. I mean, that's kind of like what this pilot is supposed to be, is try to get something close to that.

## @speaker_00 [00:57:48 → 00:57:48]

Yeah.

## @speaker_04 [00:57:49 → 00:58:10]

Do you... I'm trying to look at these questions, see if they're actually useful. Talking about that, are there any specific cross-service interactions that are really fragile or painful right now?

## @speaker_00 [00:58:12 → 00:58:23]

I think today... Yeah, let me switch to the Miro.

## @speaker_01 [00:58:27 → 00:58:38]

I think the most challenging part or painful is kind of like the single store materialization.

## @speaker_00 [00:58:41 → 00:58:44]

I'm not sure whether we can

## @speaker_01 [00:58:45 → 00:59:57]

improve it in the digital trees project. So basically, as you can see here, after the segment is created by USB, the segment is rejected in catalog side. Then after that, all subsequent steps don't have any relation with USB at all. As you can see here, no any line. any line point to USB, right? So basically the DSM is kind of like a center, right, to handling the interactions with other components, right, as you can see here. So yeah, that bring a lot of inconvenience to USB. Oh, when a segment is materialized failed, we never know what's the root cause. We have to check with that team, the DSM team, oh, what's the root cause?

## @speaker_00 [00:59:59 → 00:59:59]

Yeah.

## @speaker_01 [01:00:01 → 01:00:31]

Because this is the... I think the most different one, right, compared with the other materialization use cases, like the Snowflake, because like the Snowflake, the USB, it's kind of essential, right, to central component to calling FATSOC or to calling second API, blah, blah, blah, blah.

## @speaker_04 [01:00:32 → 01:00:33]

Okay.

## @speaker_01 [01:00:33 → 01:00:33]

Yeah.

## @speaker_04 [01:00:34 → 01:00:40]

But this is, so that's single store use case, you said?

## @speaker_01 [01:00:40 → 01:00:47]

Yeah, yeah. This is the red one is a single store use case. Yeah.

## @speaker_04 [01:00:47 → 01:00:56]

And I'm not sure how much just the testing stuff will help there, because it sounds like it may be an architecture thing, if you're not getting any notifications or callbacks or whatever.

## @speaker_00 [01:00:58 → 01:00:58]

Yeah.

## @speaker_04 [01:00:59 → 01:01:15]

OK. Let's talk a little bit. Oh, actually, we're over time. If you need to go, you can. Otherwise, I have a few more questions, maybe.

## @speaker_01 [01:01:17 → 01:01:57]

Yeah, I think... We can continue our discussions maybe in the next 10 minutes. Is that okay with you? It's already 11.35 p.m. Is there a better time that works for you? Uh, I think, uh, maybe PST, uh, 5. Yeah. Uh, 4 30 PM. Yeah. Yeah. Yeah. The comfortable time for me. Yeah.

## @speaker_04 [01:01:58 → 01:01:59]

Is that when you start working?

## @speaker_01 [01:02:02 → 01:02:26]

Yeah, I can, because the 4.30 p.m. PST is 7.30 a.m. in my time zone. Okay. Yeah, but I still, it's a comfortable time for me. Yeah, I can join the solution.

## @speaker_00 [01:02:26 → 01:02:27]

Okay.

## @speaker_01 [01:02:27 → 01:03:02]

Yeah, yeah. That makes sense. Yeah, I think the time, this meeting time, right? This meeting time is also okay. Maybe in the next weeks or next upcoming days, we can still schedule the meeting at the same time today. Yeah. I might not join all meetings. But I would definitely invite, ensure at least one USB engineer came on the meeting to support you. Okay.

## @speaker_04 [01:03:03 → 01:03:09]

I'm just trying to... I'm still waiting on getting Slack access and some other things like that. Oh, okay.

## @speaker_01 [01:03:10 → 01:03:11]

You don't have Slack access at all?

## @speaker_04 [01:03:12 → 01:03:15]

Not yet. But I think this week I'll probably get it.

## @speaker_01 [01:03:16 → 01:03:18]

Okay. That's good. Yeah.

## @speaker_04 [01:03:19 → 01:03:30]

But let me ask, so if we can mock out one of the dependencies that USB talks to and let you run tests on it locally in seconds, which dependency would matter the most?

## @speaker_00 [01:03:34 → 01:04:18]

The dependency, I think, let me think about it. Segment equalization. Segment equation. This change is not very important. Maybe the materialization, but from your opinion, right?

## @speaker_01 [01:04:20 → 01:04:29]

It's Snowflake, right? This kind of use case is the best one, right?

## @speaker_04 [01:04:30 → 01:04:32]

You think it's the best one to mock out?

## @speaker_01 [01:04:32 → 01:05:48]

Yeah. No, no, no. I'm not sure. I'm asking you, right? Because here, USB, after the segment is created in cataloger by USB, right? Then there is no any... any interactions with usb at all so uh the downstream right the service uh they are uh they are interacting with catalog uh and the sucking circle right right then eventually uh dsm will update the DSM will update the segment call. DSM will update the segment call. After we save the call, segment call will update with catalog for the segment status from updating to ready. But no matter what happening at the downstream, they don't have any relationship with USB at all. So from your opinion, whether this use case could be used into the POC as the first one?

## @speaker_04 [01:05:49 → 01:05:59]

I mean, maybe. So I'm having trouble remembering, because do you own Segment Core?

## @speaker_01 [01:06:00 → 01:06:08]

No, I don't own the core. Yeah, I just own this part.

## @speaker_00 [01:06:09 → 01:06:11]

Yeah, OK. And this one.

## @speaker_01 [01:06:12 → 01:06:21]

But as you can see here, same engine doesn't have any relationship with other components in this case, right? The single store use case.

## @speaker_04 [01:06:22 → 01:06:23]

So you're just?

## @speaker_01 [01:06:24 → 01:06:26]

I just own this one, the USB one.

## @speaker_04 [01:06:27 → 01:06:34]

What's the input? Is it through the UI, somebody creates a segment, and then it goes to SPU?

## @speaker_01 [01:06:34 → 01:06:41]

Yeah, yeah, yeah. Yeah, yeah. Let me go back to the segment creation. Yeah, this is the one.

## @speaker_04 [01:06:43 → 01:06:48]

So does it go through this flow and then through the next flow? Or is it just do the normal one? Yeah, yeah.

## @speaker_01 [01:06:48 → 01:06:54]

This is segment creation. After segment is created, then we have this flow. We have this flow.

## @speaker_04 [01:06:56 → 01:07:00]

And, but this kicks off because you publish an event to PubSub?

## @speaker_01 [01:07:02 → 01:07:46]

Because we are not, USB doesn't publish the event. We call the segment API, we call the, we call catalog API to pass the segment of building metadata, right? the number six, right? After the number six is completed, right? That means the segment assert is registered in catalog site. Then catalog publish a segment equation event message. Yeah, yeah. Then the DSM receive the message from catalog then start the virtualization flow.

## @speaker_04 [01:07:47 → 01:07:53]

So it sounds like if you had a mock service, it would be the catalog service to cover this. Is that right?

## @speaker_01 [01:07:54 → 01:08:01]

The catalog service, I'm not sure. Yeah, I'm not.

## @speaker_04 [01:08:02 → 01:08:05]

Because it's the last step in the previous thing.

## @speaker_01 [01:08:05 → 01:08:26]

Yeah, the last step of the previous. Yeah. Do you think if it was a single-story use case, it's a very good one?

## @speaker_04 [01:08:28 → 01:08:39]

I'm sure. I'm still trying to understand the boundaries of the system. Because looking at the graph, nothing connects. But if you look at the previous graph, it connects to a lot of things.

## @speaker_01 [01:08:40 → 01:08:51]

Yeah, the previous graph is this one. Yeah, I think this one is kind of same equation.

## @speaker_00 [01:08:52 → 01:08:58]

Yeah, same equation for single star and the snowflake. Yeah.

## @speaker_04 [01:08:59 → 01:09:36]

But I also might need to talk more in the future about how this works to solidify how it works up here. But when I'm talking about testing and mock services, I'm thinking about a mock service for Segment Core, a mock service for Catalog, a mock service for whatever that is on the left side. Oh, DSM. Like if you could have mocks for those that you could run really fast locally and run tests against, would that be valuable? Or is it not that valuable right now?

> [pause: 5s]

## @speaker_01 [01:09:41 → 01:10:43]

Yeah, I think the mock, because in our past experience, right, we didn't encounter too much problems in those interactions, right, with those dependency services. The most challenging or the area we, the area we encountered most questions is in the segment of materialization, right? Sometimes we, when we encounter the segment get stuck, right? Get stuck means the segment is always, always is updating, right? Never changed to ready, means the segment never be materialized. We have no idea.

## @speaker_04 [01:10:43 → 01:10:55]

I mean, is there a way that testing can solve that? Because I keep thinking that sounds like an architecture problem.

## @speaker_00 [01:10:59 → 01:11:07]

Yeah, I don't know. OK.

## @speaker_01 [01:11:09 → 01:13:19]

I think probably we can try from the segment creation first. Although the segment creation is not the one we encounter too many problems in the past. But we can pick each. for the one, right, in the POC. Because like in the materialization, right, I think the, I don't know how to, from USB perspective, how do we, maybe we need to check with the DSM team, right, whether, I mean the involved DSM team, oh, whether they can integrate with some work service, right, to quickly to virtualize the segment, right? You know, by integrating the mock service, like the mock service of Arcane Circle, mock service of the same call segment and the catalog API, right? then any request from the USB side, or any segment created in catalog side by USB, they can quickly change the segment status to ready, means the segment is materialized. But in the reality, DSM just calling some mock service to materialize the segment. Yeah, I think I need to have internal discussions with my team. Maybe they have other suggestions. Yeah. By the way, how do I contact with you?

## @speaker_04 [01:13:19 → 01:13:22]

Right now, it's through email. They're trying to get me on Slack.

## @speaker_01 [01:13:22 → 01:13:27]

Oh, okay. Okay, okay, yeah. Okay. Yeah, yeah.

## @speaker_04 [01:13:28 → 01:13:29]

There's my email.

## @speaker_01 [01:13:31 → 01:13:33]

Okay, let me save it. Then I can.

## @speaker_00 [01:13:35 → 01:13:39]

Okay, yeah. Let me save it. OK.

## @speaker_unknown [01:13:40 → 01:13:40]

Cool.

## @speaker_04 [01:13:41 → 01:14:02]

All right. Yeah, OK. Since it's so late, that's probably all the most valuable questions I have. I need to meet with another team later this week, and then I'll try to dig through these. If I have any, actually, what is your email? I don't know.

## @speaker_00 [01:14:02 → 01:14:04]

Let me share. Yeah.

## @speaker_04 [01:14:04 → 01:14:06]

Oh, I should have the calendar invite.

## @speaker_00 [01:14:09 → 01:14:13]

Yeah, let me send in the Zoom chat. Yeah, just send in the Zoom chat.

## @speaker_unknown [01:14:15 → 01:14:16]

Thank you.

## @speaker_01 [01:14:17 → 01:14:22]

Yeah, any questions, we can just leverage the email to discuss for the time being.

## @speaker_04 [01:14:24 → 01:14:30]

Yeah, awesome. All right, well, thank you very much for taking the time super late at night to go over.

## @speaker_01 [01:14:30 → 01:14:41]

Oh, yeah, thank you. Thank you. Yeah, thank you for joining us and supporting us. very happy to work with you in the next several weeks.

## @speaker_04 [01:14:42 → 01:14:49]

I hope I can provide some value and hopefully solve some problems. Thank you. Have a great night.

## @speaker_00 [01:14:51 → 01:14:52]

Have a good day. Bye-bye.
