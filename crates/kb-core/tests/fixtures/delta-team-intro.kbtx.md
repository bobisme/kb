---
type: source
title: 2026-04-09 LiveRamp Delta Team Intro Transcript
recording_date: 2026-04-09
audio_file: GMT20260409-143115_Recording.m4a
duration_seconds: 3615.4
speakers_detected: 6
asr_model: large-v3
diarization_model: pyannote/speaker-diarization-community-1
language: en
---

## Speakers

- @speaker_00: unknown
- @speaker_02: unknown
- @speaker_03: unknown
- @speaker_01: unknown
- @speaker_unknown: unknown
- @speaker_04: unknown

# Transcript

## @speaker_00 [00:00:00 → 00:00:03]

Shannon mentioned somebody was going to join.

## @speaker_02 [00:00:04 → 00:00:08]

Conor. Conor will join us. I don't know if he's on there.

## @speaker_03 [00:00:11 → 00:00:12]

Conor will join today's meeting?

## @speaker_02 [00:00:14 → 00:00:16]

Yeah, he had a Conor. I don't know.

## @speaker_03 [00:00:16 → 00:00:21]

Let me double try. Did we invite him?

## @speaker_02 [00:00:22 → 00:00:23]

Yeah, same on the invite.

> [pause: 6s]

## @speaker_03 [00:00:29 → 00:00:31]

Oh, we didn't invite him.

## @speaker_02 [00:00:33 → 00:00:35]

Did you know? I can see him on the invite.

## @speaker_03 [00:00:37 → 00:00:38]

Really?

## @speaker_02 [00:00:39 → 00:00:43]

Yeah, he is on the invite. Yeah, I can see that.

## @speaker_03 [00:00:45 → 00:00:46]

Oh, Connor.

## @speaker_02 [00:00:47 → 00:01:05]

Yeah, Connor. Oh, good. But yeah, I mean, if you're happy to, we can kick off with like, maybe Bob, if you just wanted to make sure we're on the same page, like, what are you hoping to get?

## @speaker_01 [00:01:05 → 00:01:19]

Or maybe that's Josh, you know, from this meeting? Sure. Have y'all talked about the, I guess the high level, like shared the proposal or talked about any of that yet? Do you know, Josh?

## @speaker_02 [00:01:23 → 00:01:27]

or anybody. Yeah, sorry, Josh, go for it.

## @speaker_03 [00:01:28 → 00:02:21]

Yeah, so Sean connected Bob with a long run pilot team to start with understanding the bike design first. And after that, Bob will connect with his team to see how we can using agent AI to achieve development tasks in our local environment that will speed up our software process, also improve quality. And all the activity we plan to achieve is before we deploy the feature PR to the QA environment. Yeah. Because we also work on a program to set up a queueing environment for the parallel team. This stage is before that happened. Yeah.

## @speaker_01 [00:02:23 → 00:02:46]

Yeah, so we're looking to do some more autonomous development using AI agents, but in order to get that, we need some strong guardrails in place, and that's beginning with specifically around sort of mocking out services and being able to run them locally on your computer or on your laptop, whatever. Okay. Yeah.

## @speaker_unknown [00:02:47 → 00:02:47]

Cool.

## @speaker_02 [00:02:47 → 00:03:40]

So yeah, I think that's exactly what Shannon explained to us. So within LiveRamp, I'm not sure if you were here when we did the acquisition for Habu. But we acquired ABU like three months ago, so we still maintain the GitHub organization. We do have to. ABU has about 20 microservices that we tend to run locally. I'll just show you some of how we do our day-to-day development. The team has been using AI for about 18 months to implement end-to-end features. So I have Carlos from my team who is, you know, one of the developers doing stuff. Hey, Shana. So, yeah, so I'll just maybe five minutes just to share my screen and then show what it's like.

## @speaker_00 [00:03:40 → 00:03:47]

I assume you've done introductions and know who Bob is.

## @speaker_02 [00:03:49 → 00:03:54]

Well, maybe you want to do that again. I think Josh did most of it.

## @speaker_00 [00:03:54 → 00:06:34]

All right. I was just going to say, so Bob used to work with LiveRamp back in the day. I think you were in security, like DevSecOps and stuff like that. It has been just like living in the agentic realm, building tools and stuff for a while. And Shankar and I and several of us have been talking about this digital twin, this live ramp in a box, this local stack concept for a while. There's been various versions of this that have been attempted at live ramp. Connor Taff is actually working on that crib thing that's actually really cool looking awesome. Bob, we'll get that to you for you to take a look at as you get Slack access today. hopefully, and stuff. But the idea here is Bob is going to be hands-on to help us build that and bring in capacity to essentially work with y'all and then also Xiao Dong to kind of focus on two areas and like, what do these two areas need to get something like that? And then, so really, I want to let him drive most of the conversations and questions and here and then i can help build any bridges we need or unlock anything we need resource wise or uh tech wise or red tape cake whatever that we need to get stuff done um just let me know and i will fight whatever political battles we need to fight so um cool yeah And then Ope and Carlos, they're awesome engineers over on the clean room side, focused on a team called Delta. So they do a lot of like cross-pollination of services on their team. So they do shared ownership and work things. But the things they're really known for that I remember is like the clean room flows system. They built that, which is like the orchestration of clean rooms and everything. And then, um, they also built like the identity bridge service and stuff that allows, um, us to bridge between like the live rent CAC identity and the, the hobby identity and, um, vice versa. So, um, and I don't know what they're working on these days, but, um, probably something cool. Um, yeah, so that's TLDRs. Um, and then I think, okay, you were just getting into fun stuff.

## @speaker_02 [00:06:35 → 00:06:37]

Yeah, yeah, yeah, yeah.

## @speaker_01 [00:06:37 → 00:06:48]

So, yeah. How do you spell Habu? I'm typing up notes, and I want to make sure I get it right. H-A-B-U. Okay, that's what I got. Cool.

## @speaker_02 [00:06:49 → 00:11:43]

Cool. So, Habu has this one GitHub organization different from LiveRamp, and I think it's going to be like that. Now we've integrated Opta since you left Shannon. So I don't think we'll be moving over into live RAM. So I think Bob will probably need to get access to both. So everything I'm sharing here is strictly in the ABU GitHub repository. So again, just to give a lay of the land, broadly speaking, ABU is control plane and data plane, really. Most of where my team sits is control plane. basically like Shannon was saying, we built an orchestration system that kind of uses many of these microservices and just runs processes that kind of uses each of them for different things. And then, um, So, um, so we have a system called flows. Um, I, I really, um, I mean, maybe it has, this goes on, we can really dive deep into any of these services and how they use and how, which ones you really need to know more details about. But generally, um, data plane is where, um, the execution happens. Um, so Abu, um, has a concept of questions and things like that, but essentially it's just, um, a query execution engine. All of what we have here on the control plane is just to put policies in place on how partners collaborate within execution of queries And then the actual data education is happening on this side. So within the claimant team, we are split out into about five different groups and each group manages like a subset of these services. So the way we tend to go about development, so I'm just going to take you to something we are just recently doing is we would, so we had to implement something called decision logic, like, you know, this quarter that just ended. And we identified that that would require changes across like about 12 repositories like this. So my team typically is Courser first. We've been exploring Cloud Code, but most of the time we work with Brownfield projects and it just works for us in terms of indexing and things like that. Cool. So we would identify what repos we need or what services we need listed into this place. We have processes in place for how we go from Epic to actually the plan and how we do implementation. We follow all that process. But I guess since this is more focused on just setting up the dev environment, I'll focus on that aspect. So as you see, a lot of things are very manual. I have many of these things running locally. Many of these services, these are Go services mostly on the control plane. I've started most of them. And then, so generally when I'm interacting, our work is typically full stack. So it usually involves some sort of UI. And then we tend to do, all of this is local. This is, I don't know if Bob, you're familiar with the Connect platform. It was probably there before you left. So basically what I've done here is I have, done something called an override where I've pointed the connect platform, which is an NSUI. That's a front-end thing. I've pointed it to use my local stack. There's something I've done here. I've just pointed it to use my local stack. So every traffic I am exploring currently is actually, let me close this. is actually everything I'm doing on this UI, connect.dev, is actually being routed by my local stack. So typically, this is how we tend to work. Everything I'm doing here is captured here. So if I were to tag any, if I was in an agent and I was doing some sort of debugging, I would do a terminal or something and get it to pipe some of these logs over as it does its exploration. So let me see. I'm not really Shannon, like I'm not, I know. So this process is currently being automated. Like you said, I think the latest person taking a stab at it is not yet Connor trying to just set up a process where you just have one, like a single line, the command that just spins up all of these things for you locally and have all of this running, including the data plan and things like that. So currently, a lot of it is manual. I believe that a lot of people, especially Connor now, is looking into automating that thing. But typically, this is how we do local development today across multiple repos.

## @speaker_00 [00:11:45 → 00:16:31]

Yeah. So some of the things to think about that we would want Bob to work with you and Connor on on this, at least from my opinionated perspective, right, is if you spin everything up. It's actually quite a lot of memory and quite a lot of local resource usage. And so there's two things that actually Connor and I were chatting yesterday in the office about. One is the original thought that we were having on, and Bob will be familiar with this, is what services do we actually need to have live to test and do what we're trying to do and what could be mocked at that time. And so like some type of smart configuration, when you launch a local stack that says, okay, these services are live services, these services are mocked. So maybe something that's like 10 or, or five, like dependencies down, those are mocked, like, yeah, I don't care about those. But I need first order, first class dependencies need to be live or something like that. And how do we balance that so that as this local stack grows and has more capabilities, we don't suck up all of our RAM to run in this thing. And there's an efficient way to do this. to the other thing for like that kind of thought process, too, is how do we generate the mocks? How do we generate and keep up to date fixtures and stuff like that for those mocks? What type of agentic pipelines and stuff could we use to do that so that if like there's a new push to a service or something or change to test suites or whatever, like all of that could get updated so that the local stack is staying consistent with what's really happening across the enterprise. And I think the, we're also would want like frost dependency things on this. So like clean rooms right now, like if I run an activation flow out of this, you can mock everything. You can run everything in the clean room side, but then you're pretty much SOL. Once it leaves that, that Habu area goes into like the connect platform to things and you can't test that. And that's, that's one big area of, of like opportunity here. that I'm thinking about is not just the Habu services for local development, but how do we enable y'all to test against the Habu to other live ramp area integrations? So that's kind of like very applicable to the local stack, everything running on your machine. And then there was the concept that Connor and I were talking about, is if we get this working really well, and like memory or something comes up issue on a machine, and we don't want to tackle that right now, we could bake this all into a really fast boot image or something. And then we have a interface that you launch a cloud VM that has like Kubernetes or whatever in it. And like 30 seconds, you've got a warm environment and then you use like WireGuard or Tailscale or something to connect to bridge your local network to that lab network, that lab VM or whatever. And then now I've got everything, but all of the heavy stuff's running on a cloud service that's got a time to live or something that dies in three hours if I don't do anything with it. but I'm able to do all of my local development like I would normally. It's just using a VPN to bridge that running VM to my local host, et cetera. And so those are the two areas I kind of want to tease out with Bob is, one, like, how do we build and, like, do, like, selective services and stuff in the local stack and, like, start getting y'all's test environment and stack connected with a test environment and stack for... a live ramp especially like the identity workflows and all those things like everything then also I do want to continue playing on this idea of like a remote warmed session and I learned yesterday as well

## @speaker_00 [00:16:32 → 00:18:17]

after Connor and I had like this massive brainstorm session on this and we were riffing ideas, um, then of course, um, Facebook was listening and I saw a post pop up that this is how meta operates. Right. And, and so like metas engineering teams, when they go to start working on a development, um, they, and this is one reason they can move really fast on AI development is is they are able to spin up a sandbox in their meta cloud in 30 seconds. It takes 30 seconds for that sandbox to boot up, and it's got Facebook, Instagram. All of the meta products are live in that sandbox. Then it bridges the VPN to their local, and then they launch cloud code, and now they can do everything. there like it clones the code it it launches the sandbox everything's there and it mounts it all and bridges everything and then they just launch cloud code and they can operate on it um and so that i was like well that's what connor and i were just chatting um so yeah that that's the wild hair brained ideas um I think that we can tease out and I guess the, the, the questions that I have are always my questions of what do we do first? What do we want to tackle first? Where, where do we want to see like first value and stuff, et cetera. So I'll shut up and stop talking now, but yeah.

## @speaker_02 [00:18:18 → 00:19:45]

Yeah, I mean, I think those are very, very valid points that address a lot of pain points that we see daily. I would say most of the team has become sort of comfortable with, especially on the control plane, just putting up these Go services like the ones they need to do minimum testing. And then you find that a lot of people actually do end-to-end tests where you actually need to execute on data plane by pushing the code to stage, which is always a problem because, I mean, you know these two days rush and everybody's worried that they have to stop getting support. So I think that is the biggest issue we have. Most people cannot do end-to-end test. Fine, I can do like isolated test of my changes on the UI, how it affects selected control plane services. But when it comes to actually running something end-to-end, you usually need the full stack. Like you need airflow, you need temporal, you need spark, things like that is where you tend to lose a lot of people. So I think that's probably the biggest pain point The idea of setting up many of these services, to be honest, I think there are a lot of people like I know on beta team, they just run a script and it just spins up all of the services for them locally. It's not really a pain point for most people. I'd say even for us, I don't make Carlos, you probably have a different experience.

## @speaker_04 [00:19:46 → 00:20:12]

Yeah, we do have a different experience. Whenever we need to run something, we need data. And that is the main blocker that we run into, which is why it's staging like it's set up because data may live in CSV files or in a cloud provider out there. And you sometimes want your data to be precisely for your feature, which may require some customization. So locally making that is not as easy. So that's one of the challenges we always face. Yeah.

> [pause: 8s]

## @speaker_01 [00:20:20 → 00:20:34]

Excuse me. Can you tell me, like, at a high level, what exactly is, like, the Delta system? I still don't have a clear idea of that. Is it a Delta system or is it Delta team?

## @speaker_02 [00:20:35 → 00:22:38]

Delta team. Within the Clearroom team that own, like, these applications, we kind of split out into, like, five teams. There's Delta. We follow this Greek alphabet. There's Alpha, Beta, Delta, I think Gamma and one of them. So kind of just split out ownership of these things into different pillars. But essentially, what we are describing is the cleanroom end-to-end, the cleanroom product. So it's just this whole section of LiveAMP offering. And then to what Shannon is saying, within Connect, you have different other LiveAMP applications, like data management, activation, segmentation, USB, and things like that. Everything we've discussed so far is within Cleanroom, how we do end-to-end testing on Cleanroom. As Shannon mentioned, a bigger problem also is when you tend to start going across, away from just Cleanroom testing, you want to do end-to-end, like actually create a segment you run a report in claim room, have that main landing segment view that create a segment out of redo activation. Being able to do that end-to-end will truly, truly, truly be a big unlock for devs. So I think that is one big area. Today, even interacting with like AAA, so the authentication and authorization team, the services they offer, I've had to mock many of those things into like a .env file. Like if it's calling this, use this instead and all of these things. So I'm actually, everything I'm doing is not actually being, it's not calling out to any real users. live ram services so um i think making all of this just easier would uh would be probably the biggest value on log being able to just do end-to-end tests so any everybody that has access to connect can comfortably within their own um local system do end-to-end tests across live and product offering and have more confidence in what they are doing okay

## @speaker_01 [00:22:40 → 00:23:09]

So you're saying everybody working with connect, I'm trying to like figure out the, uh, so clearly that would be valuable. I'm thinking about, so we have like a three week proof of concept time range that we're looking at, at least for my initial engagement. And if like the Delta team was selected for that, like what's actually viable within that, that scope is like an end to end for connect. Sounds amazing. But like, is that plausible?

## @speaker_02 [00:23:12 → 00:23:50]

Um, that's probably far fetched. Um, we could, I think we could do something, maybe take a feature where you deploy something and, um, like you run something because we, we have a product called flow that cuts across everything playroom does. So say you were to do a feature with that, like locally, and then have a report that gets dropped into segments, um, build a pocket or whatever, and then build a segment of that. Or doing all that locally will probably be a big unlock. I mean, you can cut out activation, maybe cut out data catalog and all of those things. Okay.

## @speaker_00 [00:23:50 → 00:26:30]

I think there's actually a really good tie-in to the Segment Builder team. So the first conversation you had, Bob, was with Xiaodong on Segment Builder, right? Cleanroom relies on the APIs that Segment Builder is building. Right? So if we're making progress on the system in Segment Builder there, that directly could be used by the crib local stack that Connor is using. And the test could be Delta Team is the first hands-on to test that local stack with the Segment Builder. So maybe... most of your, your, your IC time might be to the segment builder team stuff. And then that unlocks some type of bridge to the local stack stuff that Delta team is using from Connor. Right. Um, and one reason, one specific reason we, we picked Delta team too, is because y'all are, um, early adopters and ahead of the pack on like the agentic coding and stuff like that as well, right? And so what we build here, we really want to tie into how do we best use like cloud code and stuff with this. In my mind, that means having really good documentation or local MCP servers or something that enable cloud code to touch And work with this local stack. So maybe, I'm just throwing a proposal out there. Maybe, Bob, like your big heavy focus is with segment builder and building that stack up. And then making sure that the system that Connor started can talk to that and work with it. And then Ope, you and team can really work on the agentic integration to that to enable, hey, I start up Cloud Code. I have this local stack. What is possible because of that? What can Cloud Code do now that it has access to these things? And what value could it unlock in a development process? That might be a good scope for the three weeks. Yeah, yeah.

## @speaker_02 [00:26:31 → 00:26:55]

Yeah, I think, yeah, it sounds like we can actually do this in parallel if we work this way. Because I already started looking into what Conard built. So we could play around with that. It doesn't stop what Bobby is doing with Segment Builder. And then we just come up with a simple use case or a feature that kind of relies on, you know, speaking with Segment API and just, you know, build it with a processor.

## @speaker_01 [00:26:58 → 00:27:05]

Should we talk more about the crib tool or should we talk about that separately?

## @speaker_02 [00:27:05 → 00:27:12]

No. Or maybe Shannon, you, you know, do you know enough to amend that?

## @speaker_00 [00:27:15 → 00:27:20]

Sorry. Um, I was answering a quick Slack, the, uh, the crib, the thing that Connor's been working on.

## @speaker_01 [00:27:20 → 00:27:21]

Yeah. Yeah.

## @speaker_00 [00:27:22 → 00:32:21]

Uh, I've got some screenshots that he shared. Let me pull that up. Um, and I can show that I can talk to, I don't have access to the code right now, but I can do this. Okay. Um, yeah. So the, this is kind of where we started. Um, So he started this project crib. It uses a K3D cluster, starts it up. And so you've got this crib up, crib build, crib work, points to stuff, pins version of artifacts, wires proxy to local mock dev instances and stuff, and then test runs into end test. And so he's kind of got an opinionated way of testing that uses scenarios and stuff. And when he started, it looks like he's got this kind of basic UI that spins up. Um, so that you can access all the different services. It's, it's, uh, spinning up behind the scenes. Um, and so he is doing the things like, um, running, uh, PG, uh, vault. He's got a local instance of like vault, um, local instance of airflow and, and temporal running, um, all the bits and infrastructure that, uh, Habu depends on. Um, And then it's also running a local Grafana so you can get all of the logins and everything. And so he's using my Mike rocks for the mock APIs. I think it's an open source package he found that does mocking of services. And so he's mocking IDAPI through that. The one thing he was working on, I don't know how far he's gotten, is... Um, like it was getting like all the UI stuff working, but I think he even has like local, um, Maven, um, stuff to like push builds to and pull from in the, the local infrastructure and everything. Um, and then here's the post about meta. And then he worked last night. Um, So he took the dagger demo I did with the DAG output stuff, and he pulled in and created a UI using React flow that shows the dependency of all the services running and everything. He did that last night and everything. So he's made pretty good progress. um, on this. I think it's a really good foundation for us to build on. And then we can, my, my thinking is we can add in like all the segment building pieces of connect into this, um, and start growing this, um, crib service. Um, it's called crib clean room in a box. Um, but, uh, that we can continue to, to build up on it, uh, I, yeah, whatever I was explaining to him what we were doing, he was like, hey, this. I was like, and we started talking, I was like, this is awesome, yes, this is a great start, right? I think it uninstitutionalizes a lot of knowledge on how this is all linking to, because I you could, Bob can pull up and point an agent at this crib repo and get a lot of answers on how things are working. Right. And if he, if we could start like wiring in the segment builder pieces to this as well, because segment builder uses Grafana, right? Like, so you've already got a Grafana instance here, like wire that in and all this other stuff. Then we start building this ecosystem and, And you know, it's local right now, but I do want to, I think for the first three weeks, Bob, like,

## @speaker_00 [00:32:22 → 00:33:28]

the scope should be like continuing to build up crib and local stack version. I do really want to ideate on this idea of a cloud environment that's like 30 seconds spin up and it spins up this whole stack within that. And then I bridge my VPN because then I can run multiple, right? And if I'm running, like trying to do parallel implementation across systems, I could run multiple environments, have multiple versions. of Claude running in, you know, multitask, multitasking, my implementations, and I'm not even like stepping on my own toes on my local machine, right. And so that something I do want to explore that would be valuable. But I think that would be maybe a phase two or something after we get the local stack stuff answered. Because then we can bake most of this into like a standard VM image or something. And we could like firecracker start that quick or something. I don't know. But yeah.

## @speaker_01 [00:33:29 → 00:33:40]

Okay, so you're suggesting continue developing or continue going down the crib path for the first three weeks, but like ideate on the cloud-hosted dev environment?

## @speaker_00 [00:33:41 → 00:34:53]

Yeah, yeah, exactly. Yeah, because I don't want to blow scope up too much, right? And we've got a great foundation started here, and if we can wire this in, maybe this saves us a lot of time to do stuff, and we can start wiring in the segment builder to the crib side, and that might be a really fast time to value for you. um for things and you you can that that concept of bridging the clean room system with a local stack running of the segment builder that would be a a huge unlock i think um And it's not just a huge unlock for the clean room teams. It would be a huge unlock for the agents and stuff we're building that need to walk across the entire live ramp product to do things, right? Like, so one of the XMI demos is like starts in the clean room and then needs to like push data and segments to the core live ramp to then go activate those segments based off of questions in the clean room. And if we have segment builder as part of the crib stack, I could do that locally with an agent to iterate on the agent as well.

## @speaker_02 [00:34:54 → 00:34:56]

Yeah.

## @speaker_01 [00:34:58 → 00:35:23]

Interesting. Okay. I mean, sorry. I had a bunch of questions about like Delta team and the services there. Does that matter as much? I mean, it's starting to sound like the focus should be on the USB team. And then later unlocking that ability with the Delta team. Does that sound right? Or should I just like go ahead with some of these questions?

> [pause: 6s]

## @speaker_00 [00:35:30 → 00:36:45]

It might be good just to have a good understanding. And I would think of it less of like the Delta team services, more of what are the clean room services and stuff that the Delta team is working with today? Because the way the teams operate on the clean room side is different. We've got like alpha, beta, gamma, delta, right? And they're cross-functional. So one quarter, they may be focused on this set of services. Another quarter, they may be focused on this set of services within the enterprise. There's no strict ownership of delta team owns XYZ, right? There's actual like... there's actually some like forcing functions to cross-pollinate stuff. So I don't even know if Delta team's allowed to work on flows anymore or if there's another team that's supposed to be working on that today or whatever to like, you know, not to like spread the knowledge, right? So that it's not just like Delta that knows it. So I think a good question to ask is, Ope, what are y'all focused on this quarter working on? and then kind of talk through those services and systems a little more in depth with Bob.

## @speaker_01 [00:36:45 → 00:36:50]

Also quarter, just start June, February and started beginning of April. Yeah.

## @speaker_02 [00:36:50 → 00:36:58]

Okay. I think second, second week in April. Yeah.

## @speaker_01 [00:36:58 → 00:37:02]

Cool. Should we, um,

## @speaker_02 [00:37:03 → 00:37:31]

So, um, I think I also had a question for you, Shannon, you mentioned, like, I'm just trying to work back from the goal of this. Like at the end of the day, I, you mentioned a goal of like someone boots up the XMI UI, run, create a measurement in the measurement plan, ask them to execute, has segments in there and then does activation all locally. So you're not really thinking about dev work. You just want to see somebody run everything end to end locally.

> [pause: 11s]

## @speaker_00 [00:37:42 → 00:39:32]

Sorry, hamster wheel running. I think that would be great because then we could build tests and end-to-end tests and stuff to do so that as we do local development, we can run scenarios, right? And we can do the thing that we're trying to do in staging, right? Run that local. Yeah. And so there's the different phases of development, right? Like when you're just working on a small feature and doing unit tests and working within that service, and then you've got to test that on a wider and then a wider and a wider, right, for stuff. And so that's where I do think it becomes valuable for the development of If Claude knows how to run all those tasks and stuff, and I ask it to build a feature into a service, and I give it the spec and everything, and you go back and forth, that then once it does that, it can start running the iteration and test loop of testing this wider and wider and wider blast radius of the change all locally. And maybe it can solve 90% of those problems just by running local test loops. So yes, while doing the end-to-end, is kind of where I was talking. I do think it starts smaller and then it grows in, in the test surface that it can do because the ultimate goal is to be able to run that end to end. And if I can do that, that means I can run any scenario underneath that. Right. Hmm. Yeah. Okay, cool.

## @speaker_01 [00:39:34 → 00:39:34]

Okay.

## @speaker_02 [00:39:34 → 00:39:42]

So, um, Bob, um, I know we have 10 minutes. Like, do you just want me to talk to some of the core ones we, we are currently working with?

## @speaker_01 [00:39:43 → 00:39:43]

Sure.

## @speaker_04 [00:39:46 → 00:39:59]

Okay. I've also sent in the chat an image that shows from the Cleanroom product, the purple boxes there are the services, and those are also the ones we'll be working on this quarter.

## @speaker_02 [00:40:00 → 00:40:35]

Okay. Okay. Thank you. So I think generally, like I started by saying, we have the control plane and data plane And then in the middle, we have some middleware services. So think of most of, in terms of language, most of these things are written in Go. Most of these things are in Scala. And then the things in the middleware are Java services. So just back to what the cleanroom itself looks like. And then I'll tie it back to what you see in the services. That's it.

> [pause: 8s]

## @speaker_01 [00:40:43 → 00:40:43]

Okay.

## @speaker_02 [00:40:44 → 00:45:13]

So everything you see in the clean room is like, it's functionally broken down into different services. So something like what clean rooms do I have access to is controlled by one service behind the scenes. And then you tend to know what you want to work on and then you tend to put up that service. So I'll talk through, like, as you see, the most important one most people use here is on IGNX. It is a microservice, but like you see, It's more like urban spoke. Many of the services go through it. That's where we tend to spend most of our time. It's becoming more like a monorail for these days. So that service essentially controls everything about the navigation of a claim room, what claims I have access to, what I see on the navbar, what questions I have access to, and things like that. And then within different aspects, like what data sets I have access to, that is controlled by something called 4-bit data connections, as you can see. So I'll share this with you. It kind of leads out like, what functional areas each of these services controls. But typically, you'd see that most people would use Uniginex 4-bit pre-image for managing credentials, obviously, to be able to use those data sets. And then finally, you would use Picamix. This is important because this is what bridges the gap between us and segmenting and segmentation team, how we create those export jobs that eventually land in the segment, the USB side of things. So just... So whatever I create, how it lands here, where segments get viewed on, this is handled by Picamix. And then, how do we bridge between the control plane and the data plane? So whenever somebody runs a query, everything lands here. So here in the middle where we have two things, we have something called Janus service, and then we have something called Pegleg. Basically they are like queuing systems. They are more like, okay, I've gotten all these queries from all of these partners. I just need to queue them and intelligently decide which one gets executed when and where, depending on priority. And then the data plane is where you find like most of the heavy lifting is done. So it's controlled by mostly the alpha and beta team. And then, so these are the services that you find there. It's mostly Scala, T-Spark, and all of these are libraries that are just important. But the most important one is T-Spark and Aboot T-Spark. So just to give you a real life, like back to what I was looking at earlier. So you'd see that we have implemented a feature called decision logic and just give me a minute. This was mostly done with AI. Let me see. Hello, do you know if we have the architecture diagram? Okay, so see, if you see this, you see that this is all how the agent has done the implementation across these different reports you have in my workspace. And we always, for every feature, every, every feature we produce, we have this living document that just goes with it. And this is how engineers tend to ideate on things about like discussions going back and forth. But as you can see, just mapping this down to the different services I showed you. We've generated all of these. So with this living document, most implementations, every time you start anything that has to do with decision logic in closed or in claim room, it will always reference this document to know what has been done. And whenever the model or something has made a new change, it will update this document to just make sure that it's up to date for subsequent ideations. So, like I said, these are the main things we use and it changes by feature to feature, but we always maintain this kind of document to just show how end to end, what services were affected and what exactly is the business objective of this particular feature. You have that. So you find that across many of the features that we've implemented. So this is actually the strategy we've been using for virtually everything. You see the same thing play out with like we did this as well last year. Yeah.

> [pause: 12s]

## @speaker_02 [00:45:25 → 00:46:56]

So, yeah, it's more or less the same concept where we just have this document where we describe exactly what we are trying to achieve, what are the scenarios, all the scenarios, everything documented, what are the input, what are the output responses, everything documented. like this is mostly humanly done. We use AI for some of the man-made diagrams, but it's something that engineers ideate on for weeks. And then once it's standard, we just give it to our model to create a plan. And then from that plan, there is like discussions and all of that. But this is what is committed into our GitHub repository because we want to have that distinct role with this particular feature. So these are big areas of the clean room. And then you see that there are documented in their own right. Again, I think we should have maybe for this one, this is just a business discussion. I think for this one, we had another document where we had like show what services and end-to-end execution of it. So yeah, I can point you to any of these documents. I think it will do you more justice when you go treat yourself. But essentially, we have about 20 to 25 microservices. More often than not, in reality, you only have to work across six, seven at the same time for a particular end-to-end feature. Cool. So I will stop sharing. I know I think Shannon had to...

## @speaker_04 [00:46:57 → 00:47:00]

dropped the AI meeting.

## @speaker_02 [00:47:01 → 00:47:06]

So yeah, Bob, does that kind of shed a bit more light on things?

## @speaker_01 [00:47:06 → 00:47:34]

Sheds, yeah, some light. I'm hoping I can get this recording because that was like really fast. Yeah, sorry. I'm looking at my other questions I had written down. So you said that you run your services locally. Yeah. Right, when you're doing development. Yes. Are they, are any of them like mock services or they're all just the real service?

## @speaker_02 [00:47:35 → 00:47:54]

They're the real service. You build them and then run them. Um, so all of them are, um, yeah, we, I think we, Carlos, we don't use any pipes on, on our side of things. So you actually have to make a change, build it and then reload it. Um, which is a pain obviously, but yeah, that's how we work today.

## @speaker_01 [00:47:55 → 00:47:55]

Okay.

> [pause: 6s]

## @speaker_02 [00:48:01 → 00:49:01]

so is it do you have any automated end-to-end tests well I think we kind of already talked about the end-to-end tests but like what's the biggest automated test that you have so basically we have we don't have end-to-end test for the platform we rely on our QE team to they build out something with I forgot what this thing we play right and they handle that for us. Within every service, we have unit tests, even though we have low coverage on some of the acquired repositories from Apple, but we are building it now, especially to take advantage of this AI tooling. Yeah, I think that's really how we do. Once you've done your dev, you deploy into staging environment. QA runs like this automated test every one hour or so. And then we do deployments to prod on Tuesdays and Thursdays, provided there are no issues that QA finds.

## @speaker_01 [00:49:03 → 00:49:15]

Interesting. Okay. So that last part, you said they run tests every hour or so, and then is there any automated thing that moves it to production?

## @speaker_02 [00:49:16 → 00:49:41]

No, that is, so you decide, okay, I've made changes to these five microservices. So on Tuesday, you create a PR for deployment. You specify what image tags should get deployed. QE would run a test end-to-end and confirm that nothing has broken before the SREs manually update those services on prod.

## @speaker_01 [00:49:42 → 00:49:44]

Does QE, they write all the tests?

## @speaker_04 [00:49:45 → 00:50:16]

yeah yeah they do how does qe know what to write they um they work with us right sorry yeah no they tend to collaborate with everyone with every engineer that's building a feature so i for example have one-to-one calls with the qa for delta and i am the one who guides them over what to test based on our commitments for the quarter okay oh may i ask a question who is the qa you reach out to Venom. That would be Venom from India team.

## @speaker_01 [00:50:18 → 00:50:21]

Interesting. How big is the QE organization?

## @speaker_02 [00:50:23 → 00:50:38]

So within Clearroom, they tend to embed one QE per team. So like Carlos said, for our team, we have a dedicated press center. I think within Clearroom, Josh, right, you assign like four QE engineers to us.

## @speaker_03 [00:50:39 → 00:50:45]

Yeah, for HAPU, we have, for total HAPU, we have, for total clean room, we have 5QE, Bob.

## @speaker_01 [00:50:46 → 00:51:07]

5QE. Interesting. Okay. Okay, here's a question that's very broad, but like, so when you deploy, when you develop something, how long does it take from saving a file to knowing if something broke?

> [pause: 5s]

## @speaker_02 [00:51:13 → 00:51:56]

I think it varies. Typically, if I'm doing something which is mostly just UI and I really don't care about, like, I know it's a change. Maybe, for instance, somebody saying, oh, this drop-down menu is not populating or it's not working. That you find out immediately, like, once you do it on your dev. But when it actually has to do with anything end-to-end test within the claim room, actual query execution, it could take hours. because basically you need to deploy that to dev environment, sorry, stage environment, and then set up the right data sets that mimics production data or mimics the scenario. So it varies widely. It could just go from a few minutes to like a few hours or potentially days and weeks.

## @speaker_01 [00:51:57 → 00:52:16]

Or potentially days and weeks. Yeah. So you talked about So you have 20 services, but it sounds like on any one given thing, you have to run six to seven, right?

## @speaker_02 [00:52:16 → 00:52:26]

Yeah. Yeah. Most of the time you'd typically do six, seven for most features. It's very unlikely that you are making a change that affects so many other things.

## @speaker_01 [00:52:31 → 00:52:42]

What percentage would you say of stuff that you develop locally that you have to deploy to staging to know whether or not it's accurate?

## @speaker_04 [00:52:46 → 00:53:04]

Yeah, I have a, I have a good, a good grasp. Um, I would say that, um, 50%. Okay. Cause, um, a lot of the times we need staging for the data. Um, so 50 to 60, that's, that's really the reason why we use staging. It's the data there.

## @speaker_01 [00:53:08 → 00:53:15]

So that makes me wonder, like, is it even tractable? Like how critical is that data?

## @speaker_04 [00:53:17 → 00:53:38]

Oh, it's for, for the, for Delta team, it's crucial because most of our features requires runs. You cannot run without data. Um, so plenty of times you can build around like the feature, the, the, the body of the feature, but you cannot really execute it until you have the right data. And that's the last test we tend to do. Like how does, how does it execute?

## @speaker_01 [00:53:40 → 00:53:49]

Okay. And is it a situation where it's like actual customer data that can't be copied locally?

## @speaker_04 [00:53:49 → 00:54:08]

No, it's the situation of actually having that data locally running somewhere in the service that can be reachable. That has been the challenge locally, which is why we rely a lot on staging. But the data on staging, it's not customer-like. It's fake information we've placed in there. But it just has the right shape that we need.

## @speaker_03 [00:54:09 → 00:54:09]

Cool.

## @speaker_02 [00:54:10 → 00:55:00]

And I think a bigger issue is that most of these data sets, the way claim remotes that your data is persisted in different clouds, BigQuery, Snowflake, Databricks. So having those credentials given to every developer locally is not something we want to do. So you typically always have to deploy to these particular stage environments to leverage those data sets. And like you said, there is the other problem where production data varies dramatically. usually from test data. So internal teams like the cross media intelligence that sits on top of always face this problem where they're running something in prod. And then we've developed something by testing of dev data, which works, but when they actually use the very rough view, like production data, it doesn't work. So that's like another layer we face. Okay.

## @speaker_01 [00:55:00 → 00:55:16]

All right. We talked about tests. We coverage. Do you know, is there like one specific thing that breaks the most often or bugs show up the most?

## @speaker_02 [00:55:16 → 00:55:42]

You mean in terms of services or just like... Yeah, so I think the way this thing has been architected, this service called OnIgenex, I showed earlier, tends to have too many dependencies and many other services depend on it. So it's a very, like, once you make a change, the chances of breaking other people's stuff is almost at 80%.

## @speaker_01 [00:55:45 → 00:55:45]

Is what percent?

## @speaker_02 [00:55:46 → 00:55:48]

It's almost at 80%. 80%, okay.

## @speaker_01 [00:55:48 → 00:55:50]

80%, yeah.

## @speaker_04 [00:55:50 → 00:56:00]

That is the core service for cleanroom entirely. So it's also the most used. So likely the one that breaks too.

## @speaker_01 [00:56:03 → 00:56:12]

Oh, okay. So would you say that that's like the most, sounds like that's the most fragile thing?

## @speaker_04 [00:56:13 → 00:56:13]

Yeah.

## @speaker_01 [00:56:14 → 00:56:35]

Okay. Uh, you don't have any mocks or stub dependencies right now. And you're mostly using cursor. Uh, is there's no like requirement to use cloud code, even though folks have access to it, right?

## @speaker_02 [00:56:36 → 00:57:04]

No, no requirement. Um, yeah, but I, I guess like, yeah, people are kind of split. Most of us started with cursor. And then I think because we, uh, I don't know, maybe because we're working on existing reports, we just kind of stick with it. I guess like people that tend to like start up on new greenfield projects probably would go with cloud code. But yeah, I mean, people, yeah, I think Carlos, you use cloud code and cursor.

## @speaker_01 [00:57:04 → 00:57:15]

I also do sometimes. Is there anything about like cursor or even cloud code that you find particularly useful or particularly frustrating?

## @speaker_02 [00:57:16 → 00:58:00]

I think, yeah, so the reason why I use Cloud Code, or I'm sure most people on my team ever use Cloud Code, is when we run out of credit on Cursor because it's expensive. But Cursor works mostly with all of us because for most developers, you want to avoid the boilerplate of setup, like You just really want something that just works. So I think that's why a lot of people just use Coursera. The workspace feature, having it look through the repos you need all in one go. The way it indexes already your repo, you can ask questions, get answers instantly. And I just find models give better performance when you use it in Coursera than when you directly use them. Interesting.

## @speaker_01 [00:58:04 → 00:58:16]

I haven't used Cursor ever. I've been using Cloud Code since the day it came out. So no, I guess I need to try it. OK, so.

## @speaker_03 [00:58:16 → 00:58:23]

But as the fans of Cursor, I strongly suggest you try Cursor, please. Yeah, yeah.

## @speaker_01 [00:58:23 → 00:58:30]

OK. I just already have the, you can't use your Cloud Max with Cursor, so I'd have to pay for that separately.

## @speaker_02 [00:58:31 → 00:58:45]

yeah i mean you could install you can install the extension so you get it like in you use it within um cursor itself in place of um cursors agents now it's 8 30.

## @speaker_01 [00:58:47 → 00:59:02]

Okay, that's a good warning. I think that answered almost all my questions. I guess the biggest thing is time-wise, does this time work if we need to collaborate? Carlos and Ope, you're both in the UK?

## @speaker_02 [00:59:03 → 00:59:09]

Yeah, I think this works. Between me and Carlos, I think we can make things work.

## @speaker_01 [00:59:11 → 00:59:36]

Okay. I mean, I'm also the contractor here, so I can try to be more flexible. I'm just working with a team in China. You guys and China makes it more difficult. All right. Well, thanks so much for going over everything. I think it was a really interesting discussion, and hopefully I'll get Slack access and we can talk through that soon, maybe today.

## @speaker_02 [00:59:37 → 00:59:57]

yeah um there's another guy on our team he's a principal engineer who is also like working on playroom ai stuff so um i could do an intro with between both of you okay yeah uh a call or slack maybe like slack once you get slack i'll just do a quick intro

## @speaker_01 [00:59:58 → 01:00:05]

Okay. Awesome. All right. Well, thanks everybody very much for the call. Thank you. Yeah.

## @speaker_02 [01:00:06 → 01:00:07]

Okay. Thank you. Bye.

## @speaker_01 [01:00:07 → 01:00:08]

Bye.
