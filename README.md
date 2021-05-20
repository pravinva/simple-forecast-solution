# Amazon Simple Forecast Solution

![](https://img.shields.io/badge/license-MIT--0-green)
![](https://img.shields.io/github/workflow/status/aws-samples/simple-forecast-solution/pytest/main)

## Installation

### One-Click Deployment

Region name | Region code | Launch
--- | --- | ---
Asia Pacific (Sydney) | ap-southeast-2 | [![Launch Stack](https://cdn.rawgit.com/buildkite/cloudformation-launch-stack-button-svg/master/launch-stack.svg)](https://ap-southeast-2.console.aws.amazon.com/cloudformation/home?region=ap-southeast-2#/stacks/quickcreate?templateUrl=https%3A%2F%2Fsfs-public.s3-ap-southeast-2.amazonaws.com%2Ftemplate.yaml&stackName=SFS-Stack)

### Linux (CentOS) CLI
```
make deploy_all
```

## Amazon SFS Security FAQs
**Q: How does the Shared Responsibility Model apply to SFS?**

Security is a shared responsibility between AWS and the customer. AWS is responsible for the "security of the cloud" and
customers are responsible for "security in the cloud". AWS provides the SFS software "as is" without warranty of any
kind. You are responsible for securing the AWS account where you choose to deploy SFS, and for securing your deployed
copy or instance of SFS, including by applying any security patches that we may release in the future. The
[Shared Responsibility Model](https://aws.amazon.com/compliance/shared-responsibility-model/) provides more information.

**Q: Should I create a new AWS account for SFS or can I deploy it in my current AWS account?**

We recommend that you create a new AWS account for SFS and use this as a dedicated AWS account for SFS. Although you
can deploy SFS alongside other workloads in the same AWS account, we don't recommend you do so. The
[AWS Well-Architected Framework Security Pillar section on AWS Account Management and Separation](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/aws-account-management-and-separation.html)
provides more information about the benefits of using a dedicated AWS account for each of your workloads.

**Q: How I can I secure my new AWS account that I have created for SFS?**

We recommend you implement the guidance contained in the [AWS Well-Architected Framework Security Pillar](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/welcome.html).
If you are new to AWS or the Well-Architected Framework, the [AWS Well-Architected Labs](https://wellarchitectedlabs.com/)
provide hands-on labs and sample code to help you learn how to secure your account.

**Q: How do I restrict access to my deployment of the SFS application?**

We built the SFS user interface using [AWS Amplify](https://aws.amazon.com/amplify/), which uses
[Amazon Cognito](https://aws.amazon.com/cognito/) for user sign-up, sign-in, and access control. We recommend that customers
disable the Cognito capability to accept new user registrations, and we have set this as the default configuration, although
customers may choose to override it after deployment. However, doing so would allow anyone to access your deployment of
SFS and any data you may have.

Cognito also supports [federation with social and enterprise identity providers](https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-identity-federation.html).
We recommend that customers who have enterprise directory service implement enterprise federation. Federation provides an
improved experience for your users, whilst leveraging the security controls that you have already implemented in your
enterprise directory service.

**Q: How do you secure my data in motion and at rest?**

The default configuration of SFS and the stack of services it depends on enforce encryption of data in motion, and encrypts
data at rest using the [AWS Key Management Service](https://aws.amazon.com/kms/). We encourage you to verify these default
configurations after you have deployed SFS. The [AWS Security Hub](https://aws.amazon.com/security-hub/) makes it easier
for customers to assess the security posture of their AWS accounts.
