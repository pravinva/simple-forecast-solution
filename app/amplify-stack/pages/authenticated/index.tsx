// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect } from "react";
import Router from "next/router";
import { ForecastingService } from "../../services/forecasting";
import _ from "lodash";
import { Preloader } from "../../components/common/preloader/Preloader";
import { UserService, UserRole } from "../../services/user";

const userService = new UserService();
const forecasingService = new ForecastingService();

// This component is wrapped in 'withAuthenticator' under the _app.tsx
// By virtue of this component rendering, this means that the user has
// authenticated successfully. We use this component as an entrypoint
// to navigate the user to the appropriate authenticated page.
const AuthHandler = () => {
  const handleUserAuthenticated = async () => {
    const role = await userService.getUserRole();

    if (role === UserRole.Standard) {
      const reports = await forecasingService.loadUserJobs();

      if (reports.length === 0) {
        Router.replace("/authenticated/upload");
      } else {
        const mostRecentReport = _.maxBy(reports, (i) => i.createdAt);
        Router.replace(`/authenticated/report?id=${encodeURIComponent(mostRecentReport.id)}`);
      }
    } else {
      Router.replace("/authenticated/admin");
    }
  };

  useEffect(() => {
    handleUserAuthenticated();
  }, []);

  return <Preloader fullScreen />;
};

export default AuthHandler;
